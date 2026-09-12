"""
AI Machine Diagnosis Assistant
--------------------------------
A Streamlit app that helps users troubleshoot common mechanical machine
problems (Pump, Bearing, Air Compressor) using a hybrid approach:
a local technical knowledge base + an LLM (via the Groq API) for
general mechanical reasoning when the knowledge base does not cover
the reported problem.

This app is a decision-support / troubleshooting assistant.
It does NOT provide a guaranteed diagnosis.
"""

import json
import re
from pathlib import Path

import requests
import streamlit as st

# =========================================================================
# CONFIGURATION (edit these to customize the app)
# =========================================================================

APP_TITLE = "AI Machine Diagnosis Assistant"
APP_TAGLINE = "AI-assisted troubleshooting support for rotating & pneumatic equipment"

KNOWLEDGE_BASE_PATH = Path(__file__).parent / "machine_manual.json"

# Groq uses an OpenAI-compatible chat completions endpoint.
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Centralized model name - change this in one place if the model is retired.
# See https://console.groq.com/docs/models for the current list of available models.
# NOTE: llama-3.3-70b-versatile was decommissioned by Groq on Aug 16, 2026.
MODEL_NAME = "openai/gpt-oss-120b"

# Minimum number of "meaningful" words required in the free-text symptom
# description before we consider it detailed enough to skip follow-up questions.
MIN_DESCRIPTION_WORDS = 6

# Minimum keyword-overlap score for a knowledge base entry to be treated
# as a genuine manual match (rather than falling back to AI reasoning).
MANUAL_MATCH_THRESHOLD = 1

MACHINE_ICONS = {
    "Pump": "🔧",
    "Bearing": "⚙️",
    "Air Compressor": "🗜️",
}

MACHINE_DESCRIPTIONS = {
    "Pump": "Centrifugal pumps used to transfer liquid in industrial systems.",
    "Bearing": "Rolling-element bearings that support rotating shafts.",
    "Air Compressor": "Machines that compress and supply pressurized air.",
}


# =========================================================================
# PAGE SETUP
# =========================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛠️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; max-width: 900px;}
    .machine-card {
        border: 1px solid #333c47;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        background-color: rgba(120,120,140,0.06);
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .machine-card:hover {
        transform: scale(1.03);
        border-color: #6c7a8f;
    }
    .machine-card .card-icon {
        font-size: 2.2rem;
    }
    .machine-card .card-title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-top: 0.3rem;
    }
    .machine-card .card-desc {
        font-size: 0.82rem;
        opacity: 0.75;
        margin-top: 0.3rem;
    }
    .source-tag-manual {
        display:inline-block; padding:2px 10px; border-radius:12px;
        background-color:#1f6f43; color:white; font-size:0.78rem; font-weight:600;
    }
    .source-tag-ai {
        display:inline-block; padding:2px 10px; border-radius:12px;
        background-color:#8a5a00; color:white; font-size:0.78rem; font-weight:600;
    }
    .safety-box {
        border-left: 4px solid #c62828;
        background-color: rgba(198,40,40,0.08);
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# KNOWLEDGE BASE LOADING
# =========================================================================

@st.cache_data(show_spinner=False)
def load_knowledge_base():
    """Load and validate the JSON knowledge base. Returns (data, error_message)."""
    try:
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not data:
            return None, "The knowledge base file is empty or not formatted correctly."
        return data, None
    except FileNotFoundError:
        return None, f"Knowledge base file not found at: {KNOWLEDGE_BASE_PATH}"
    except json.JSONDecodeError as e:
        return None, f"The knowledge base file contains invalid JSON: {e}"
    except Exception as e:
        return None, f"Could not load the knowledge base: {e}"


def get_common_problems(kb, machine):
    """Return the list of predefined common problems for a machine, plus 'Other'."""
    problems = list(kb.get(machine, {}).get("common_problems", []))
    problems.append("Other")
    return problems


# =========================================================================
# RETRIEVAL LOGIC (lightweight keyword-based matching - no extra API calls)
# =========================================================================

def _tokenize(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def retrieve_manual_match(kb, machine, problem, description):
    """
    Look for the knowledge base entry that best matches the selected problem
    and/or the free-text description for the given machine.

    Returns (entry_dict_or_None, match_score).
    """
    machine_data = kb.get(machine, {})
    entries = machine_data.get("entries", [])
    if not entries:
        return None, 0

    query_tokens = _tokenize(f"{problem} {description}")
    if problem and problem != "Other":
        query_tokens |= _tokenize(problem)

    best_entry = None
    best_score = 0

    for entry in entries:
        entry_tokens = _tokenize(entry.get("problem", ""))
        for kw in entry.get("keywords", []):
            entry_tokens |= _tokenize(kw)

        score = len(query_tokens & entry_tokens)

        # Exact / near-exact match on the predefined problem name is a strong signal.
        if problem and problem != "Other" and problem.strip().lower() == entry.get("problem", "").strip().lower():
            score += 3

        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= MANUAL_MATCH_THRESHOLD:
        return best_entry, best_score
    return None, best_score


# =========================================================================
# INPUT SUFFICIENCY CHECK (follow-up questions)
# =========================================================================

FOLLOW_UP_QUESTIONS = [
    "When did the problem start?",
    "Is the machine producing any unusual noise?",
    "Is there excessive vibration?",
    "Is there any visible leakage (oil, air, or liquid)?",
    "Does the problem happen continuously or intermittently?",
    "Has machine performance changed recently?",
]


def is_input_sufficient(problem, description):
    """
    Decide whether we have enough information to attempt a diagnosis,
    or whether we should ask follow-up questions first.
    """
    description = (description or "").strip()

    if problem == "Other":
        # "Other" relies entirely on the free-text description.
        word_count = len(description.split())
        return word_count >= MIN_DESCRIPTION_WORDS

    # A predefined problem was selected - that alone is enough to attempt
    # a diagnosis, even with a short/empty description.
    return True


# =========================================================================
# LLM (GROQ API) INTEGRATION
# =========================================================================

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def build_system_prompt():
    return (
        "You are an experienced mechanical troubleshooting assistant helping a user "
        "diagnose a possible problem with an industrial machine (pump, bearing, or air "
        "compressor). You are a decision-support tool, not a guaranteed diagnosis. "
        "Reason from the reported symptoms as a whole rather than matching keywords in "
        "isolation, and consider how the reported symptoms may relate to a common "
        "underlying cause. Never invent symptoms or observations the user did not report. "
        "\n\n"
        "You will be given the machine type, the reported problem/symptoms, and, when "
        "available, relevant excerpts from a technical manual. "
        "\n\n"
        "Rules:\n"
        "1. If manual excerpts are provided and they are relevant, base your possible "
        "causes primarily on them and label those findings 'Manual-Based Finding'.\n"
        "2. If the manual excerpts are missing, insufficient, or not relevant to the "
        "reported problem, use general mechanical engineering knowledge and label those "
        "findings 'AI-Based Possibility'. Never claim an AI-generated possibility came "
        "from the manual.\n"
        "3. List only as many possible causes as are technically relevant (do not pad "
        "the list). Rank them under the headings 'Most Likely', 'Possible', and "
        "'Less Likely' (omit any heading that has no entries).\n"
        "4. For each possible cause, give: the cause, the reasoning/evidence tying it to "
        "the reported symptoms, and its source label.\n"
        "5. Provide practical, concise, sequential troubleshooting steps. Do not provide "
        "detailed repair instructions or anything unsafe.\n"
        "6. Always include a safety warning appropriate to the machine and symptoms "
        "(isolation/shutdown before inspection, PPE, and recommending a qualified "
        "technician when the issue requires specialist repair).\n"
        "7. State an overall confidence level (High / Medium / Low) based on how much "
        "evidence is available. Never invent a numeric percentage.\n"
        "8. Respond ONLY in the following Markdown structure, with no extra preamble:\n\n"
        "## Diagnosis Summary\n"
        "(2-4 sentence plain-language summary)\n\n"
        "## Possible Issues\n"
        "### Most Likely\n"
        "**Possible Cause:** ...\n"
        "**Why:** ...\n"
        "**Source:** Manual-Based Finding / AI-Based Possibility\n"
        "(repeat for each item under this heading; then repeat the same pattern for "
        "'### Possible' and '### Less Likely' if relevant)\n\n"
        "## Suggested Troubleshooting Steps\n"
        "1. ...\n\n"
        "## Recommended Next Check\n"
        "(one clear next inspection/check)\n\n"
        "## Confidence\n"
        "High / Medium / Low - (one short sentence why)\n\n"
        "## Safety Warning\n"
        "(relevant safety message)"
    )


def build_user_prompt(machine, problem, description, matched_entry):
    lines = [f"Machine: {machine}"]
    lines.append(f"Reported problem: {problem}")
    if description.strip():
        lines.append(f"User-described symptoms: {description.strip()}")

    if matched_entry:
        lines.append("\nRelevant manual excerpt (use this as the primary source where applicable):")
        lines.append(f"- Manual problem entry: {matched_entry.get('problem')}")
        lines.append(f"- Possible causes in manual: {', '.join(matched_entry.get('possible_causes', []))}")
        lines.append(f"- Manual troubleshooting steps: {', '.join(matched_entry.get('troubleshooting', []))}")
        lines.append(f"- Manual suggested solutions: {', '.join(matched_entry.get('solutions', []))}")
    else:
        lines.append(
            "\nNo directly relevant manual entry was found for this problem. "
            "Use general mechanical engineering reasoning and label findings as "
            "'AI-Based Possibility'."
        )

    return "\n".join(lines)


def call_groq_api(system_prompt, user_prompt, api_key):
    """
    Call the Groq chat completions endpoint.
    Returns (response_text_or_None, error_message_or_None).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1200,
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    except requests.exceptions.Timeout:
        return None, "The request to the AI service timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the AI service. Please check your internet connection."
    except requests.exceptions.RequestException as e:
        return None, f"An unexpected network error occurred: {e}"

    if response.status_code == 401:
        return None, "The API key was rejected. Please check your GROQ_API_KEY in Streamlit secrets."
    if response.status_code == 429:
        return None, "The free-tier rate limit has been reached. Please wait a moment and try again."
    if response.status_code >= 500:
        return None, "The AI service is currently unavailable. Please try again shortly."
    if response.status_code != 200:
        return None, f"The AI service returned an error (status {response.status_code}). Please try again."

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError):
        return None, "The AI service returned an unexpected or empty response. Please try again."

    if not content or not content.strip():
        return None, "The AI service returned an empty response. Please try again."

    return content.strip(), None


# =========================================================================
# UI HELPERS
# =========================================================================

def go_to(page, **kwargs):
    st.session_state.page = page
    for k, v in kwargs.items():
        st.session_state[k] = v


def init_session_state():
    defaults = {
        "page": "home",
        "selected_machine": None,
        "selected_problem": None,
        "description": "",
        "report": None,
        "matched_entry": None,
        "awaiting_more_info": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_header():
    st.title(f"🛠️ {APP_TITLE}")
    st.caption(APP_TAGLINE)
    st.info(
        "This tool suggests **possible** causes and troubleshooting steps based on a "
        "technical knowledge base and AI reasoning. It is not a certified diagnosis. "
        "For critical or unsafe conditions, consult a qualified technician.",
        icon="ℹ️",
    )


# =========================================================================
# PAGES
# =========================================================================

def page_home(kb):
    render_header()
    st.subheader("Select a machine to begin")

    cols = st.columns(3)
    machines = ["Pump", "Bearing", "Air Compressor"]
    for col, machine in zip(cols, machines):
        with col:
            st.markdown(
                f"""
                <div class="machine-card">
                    <div class="card-icon">{MACHINE_ICONS[machine]}</div>
                    <div class="card-title">{machine}</div>
                    <div class="card-desc">{MACHINE_DESCRIPTIONS[machine]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            if st.button("Start Diagnosis", key=f"start_{machine}", use_container_width=True):
                go_to(
                    "diagnose",
                    selected_machine=machine,
                    selected_problem=None,
                    description="",
                    report=None,
                    matched_entry=None,
                    awaiting_more_info=False,
                )
                st.rerun()


def page_diagnose(kb):
    machine = st.session_state.selected_machine
    render_header()

    if st.button("← Back to machine selection"):
        go_to("home")
        st.rerun()

    st.subheader(f"{MACHINE_ICONS.get(machine, '')} {machine} Diagnosis")

    problems = get_common_problems(kb, machine)
    default_index = problems.index(st.session_state.selected_problem) if st.session_state.selected_problem in problems else 0

    problem = st.radio("Select the problem that best matches what you're seeing:", problems, index=default_index)

    description_label = (
        "Describe the problem or symptoms you are experiencing:"
        if problem == "Other"
        else "Add any additional details about the symptoms (optional but helpful):"
    )
    description = st.text_area(description_label, value=st.session_state.description, height=120,
                                placeholder="e.g. The pump starts normally but stops after about 10 minutes.")

    diagnose_clicked = st.button("🔍 Run Diagnosis", type="primary", use_container_width=True)

    if diagnose_clicked:
        st.session_state.selected_problem = problem
        st.session_state.description = description

        if problem == "Other" and not description.strip():
            st.warning("Please describe the problem or symptoms before running the diagnosis.")
            return

        if not is_input_sufficient(problem, description):
            st.session_state.awaiting_more_info = True
            st.session_state.report = None
        else:
            st.session_state.awaiting_more_info = False
            run_diagnosis(kb, machine, problem, description)

    if st.session_state.awaiting_more_info:
        render_follow_up_questions()

    if st.session_state.report:
        render_report(machine, st.session_state.selected_problem, st.session_state.matched_entry, st.session_state.report)


def render_follow_up_questions():
    st.warning("The description is a bit short to give a reliable assessment. Please answer a few quick questions, or add more detail above and run the diagnosis again.")
    with st.expander("Helpful questions to answer in your description", expanded=True):
        for q in FOLLOW_UP_QUESTIONS:
            st.markdown(f"- {q}")


def run_diagnosis(kb, machine, problem, description):
    api_key = get_api_key()
    if not api_key:
        st.error(
            "No Groq API key was found. Add `GROQ_API_KEY` to your Streamlit secrets "
            "(see README.md) to enable AI diagnosis."
        )
        return

    matched_entry, score = retrieve_manual_match(kb, machine, problem, description)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(machine, problem, description, matched_entry)

    with st.spinner("Analyzing symptoms and generating diagnosis..."):
        content, error = call_groq_api(system_prompt, user_prompt, api_key)

    if error:
        st.error(error)
        return

    st.session_state.report = content
    st.session_state.matched_entry = matched_entry


def render_report(machine, problem, matched_entry, report_text):
    st.divider()
    st.subheader("📋 Diagnosis Report")

    source_html = (
        '<span class="source-tag-manual">Manual-supported problem</span>'
        if matched_entry
        else '<span class="source-tag-ai">Primarily AI reasoning</span>'
    )
    st.markdown(f"**Machine:** {machine} &nbsp;&nbsp;|&nbsp;&nbsp; **Reported problem:** {problem} &nbsp;&nbsp;|&nbsp;&nbsp; {source_html}", unsafe_allow_html=True)
    st.write("")

    st.markdown(report_text)

    if matched_entry:
        with st.expander("View matched manual excerpt"):
            st.markdown(f"**Manual problem entry:** {matched_entry.get('problem')}")
            st.markdown("**Possible causes (manual):**")
            for c in matched_entry.get("possible_causes", []):
                st.markdown(f"- {c}")
            st.markdown("**Manual troubleshooting steps:**")
            for t in matched_entry.get("troubleshooting", []):
                st.markdown(f"- {t}")
            st.markdown("**Manual suggested solutions:**")
            for s in matched_entry.get("solutions", []):
                st.markdown(f"- {s}")

    st.markdown('<div class="safety-box">⚠️ Always shut down and isolate the machine safely before physical inspection. If the problem persists or requires specialist repair, contact a qualified technician.</div>', unsafe_allow_html=True)


# =========================================================================
# MAIN
# =========================================================================

def main():
    init_session_state()

    kb, kb_error = load_knowledge_base()
    if kb_error:
        st.title(f"🛠️ {APP_TITLE}")
        st.error(f"Could not load the technical knowledge base: {kb_error}")
        st.stop()

    if st.session_state.page == "home":
        page_home(kb)
    else:
        page_diagnose(kb)


if __name__ == "__main__":
    main()
