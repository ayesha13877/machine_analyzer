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

import html
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

# Per-machine pastel tint (card icon + glow). Purely visual.
MACHINE_TINTS = {
    "Pump": {"tint": "#D6E1FF", "glow": "rgba(115,140,255,0.55)"},
    "Bearing": {"tint": "#E4DAFF", "glow": "rgba(160,135,255,0.55)"},
    "Air Compressor": {"tint": "#CFF5E6", "glow": "rgba(90,205,160,0.55)"},
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

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=Figtree:wght@400;500;600;700&display=swap');

:root {
    --ink: #1F2340;
    --muted: #5F6690;
    --blue: #8FA8FF;
    --lav: #C3B4FF;
    --mint: #A8EFD3;
    --glass: rgba(255,255,255,0.58);
    --glass-strong: rgba(255,255,255,0.78);
    --edge: rgba(255,255,255,0.85);
    --shadow: 0 1px 2px rgba(60,70,140,0.06), 0 14px 34px -14px rgba(80,90,190,0.28);
    --ease: cubic-bezier(0.2, 0.7, 0.2, 1);
}

/* ---------- Base + layered background ---------- */
.stApp {
    background: linear-gradient(160deg, #F6F8FF 0%, #F1EFFF 48%, #ECF9F5 100%);
    color: var(--ink);
    font-family: 'Figtree', system-ui, sans-serif;
}
.stApp::before {
    content: "";
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(520px 420px at 6% 4%, rgba(143,168,255,0.50), transparent 70%),
        radial-gradient(460px 400px at 96% 10%, rgba(195,180,255,0.50), transparent 70%);
    filter: blur(8px);
}
.stApp::after {
    content: "";
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background:
        radial-gradient(520px 440px at 4% 98%, rgba(168,239,211,0.55), transparent 70%),
        radial-gradient(420px 380px at 92% 92%, rgba(195,180,255,0.35), transparent 70%);
    filter: blur(8px);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container {
    position: relative; z-index: 1;
    padding-top: 2.6rem; padding-bottom: 4rem; max-width: 980px;
}

/* ---------- Typography ---------- */
.stApp p, .stApp li, .stApp label, .stApp textarea, .stApp button {
    font-family: 'Figtree', system-ui, sans-serif;
}
.stApp h1, .stApp h2, .stApp h3, .hero-title, .section-title, .mh-title {
    font-family: 'Bricolage Grotesque', 'Figtree', sans-serif;
    color: var(--ink);
    letter-spacing: -0.02em;
}
.stApp p, .stApp li { color: var(--ink); }

/* ---------- Hero ---------- */
.hero { position: relative; padding: 0.4rem 0 1.6rem; animation: rise 0.8s var(--ease) both; }
.hero-ring {
    position: absolute; right: -30px; top: -34px; width: 170px; height: 170px; border-radius: 50%;
    border: 1.5px solid rgba(143,168,255,0.45);
    background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.8), rgba(195,180,255,0.18) 60%, transparent 72%);
    pointer-events: none;
}
.hero-ring::after {
    content: ""; position: absolute; left: -26px; bottom: 6px; width: 30px; height: 30px; border-radius: 50%;
    background: linear-gradient(140deg, var(--mint), var(--blue)); opacity: 0.7;
    box-shadow: 0 8px 20px -6px rgba(90,205,160,0.6);
}
.pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.32rem 0.85rem; border-radius: 999px;
    background: var(--glass-strong); border: 1px solid var(--edge);
    box-shadow: var(--shadow); font-size: 0.82rem; font-weight: 600; color: var(--muted);
}
.pill .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: linear-gradient(140deg, #5FD9B0, #7C93FF);
    box-shadow: 0 0 0 4px rgba(95,217,176,0.22);
}
.hero-title {
    font-size: clamp(2.1rem, 5.4vw, 3.3rem); font-weight: 700; line-height: 1.04;
    margin: 1rem 0 0.7rem; max-width: 640px;
    background: linear-gradient(115deg, #1F2340 0%, #3E4BC8 55%, #8A69F0 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
}
.hero-sub { color: var(--muted); font-size: 1.05rem; max-width: 520px; line-height: 1.55; margin: 0; }

.feature-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; margin: 1.6rem 0 0.4rem; }
.feature {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.75rem 0.9rem; border-radius: 18px;
    background: var(--glass); border: 1px solid var(--edge);
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}
.feature .f-icon {
    width: 36px; height: 36px; border-radius: 12px; display: grid; place-items: center; font-size: 1.05rem;
    background: linear-gradient(145deg, #fff, var(--tint, #E4DAFF));
    box-shadow: 0 6px 14px -8px var(--glow, rgba(120,130,255,0.6));
    flex-shrink: 0;
}
.feature b { display: block; font-size: 0.9rem; color: var(--ink); }
.feature span { font-size: 0.78rem; color: var(--muted); }
@media (max-width: 720px) { .feature-row { grid-template-columns: 1fr; } .hero-ring { display: none; } }

.notice {
    margin: 1rem 0 0.4rem; padding: 0.8rem 1.1rem; border-radius: 16px;
    background: rgba(255,255,255,0.5); border: 1px dashed rgba(124,147,255,0.4);
    font-size: 0.88rem; color: var(--muted); line-height: 1.5;
}
.notice b { color: var(--ink); }

/* ---------- Compact header (diagnose page) ---------- */
.brand { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.4rem; }
.brand .logo {
    width: 38px; height: 38px; border-radius: 13px; display: grid; place-items: center; font-size: 1.1rem;
    background: linear-gradient(145deg, #fff, #DCE4FF); box-shadow: 0 8px 18px -8px rgba(115,140,255,0.7);
}
.brand .name { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.05rem; color: var(--ink); }

.machine-head { display: flex; align-items: center; gap: 1rem; margin: 1.2rem 0 1.4rem; animation: rise 0.6s var(--ease) both; }
.mh-title { font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.mh-desc { color: var(--muted); font-size: 0.95rem; margin-top: 0.2rem; }

.section-title { font-size: 1.15rem; font-weight: 700; margin: 0.6rem 0 0.15rem; }
.section-sub { color: var(--muted); font-size: 0.9rem; margin-bottom: 0.7rem; }

/* ---------- Machine cards ---------- */
.machine-card {
    position: relative; overflow: hidden;
    border-radius: 26px; padding: 1.6rem 1.2rem 1.4rem; min-height: 232px;
    display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;
    background: var(--glass); border: 1px solid var(--edge);
    backdrop-filter: blur(18px) saturate(140%); -webkit-backdrop-filter: blur(18px) saturate(140%);
    box-shadow: var(--shadow);
    transition: transform 0.35s var(--ease), box-shadow 0.35s var(--ease);
}
.machine-card::before {
    content: ""; position: absolute; width: 170px; height: 170px; border-radius: 50%;
    top: -75px; right: -65px; opacity: 0.6;
    background: radial-gradient(circle, var(--tint) 0%, transparent 70%);
}
.machine-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 2px 4px rgba(60,70,140,0.06), 0 26px 50px -18px var(--glow);
}
.machine-card .card-icon {
    position: relative; width: 68px; height: 68px; border-radius: 22px; display: grid; place-items: center; font-size: 1.9rem;
    background: linear-gradient(145deg, rgba(255,255,255,0.98), var(--tint));
    box-shadow: 0 12px 26px -12px var(--glow), inset 0 1px 0 #fff;
    transition: transform 0.35s var(--ease);
}
.machine-card:hover .card-icon { transform: rotate(-6deg) scale(1.06); }
.machine-card .card-title { position: relative; font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.2rem; margin-top: 0.9rem; color: var(--ink); }
.machine-card .card-desc { position: relative; font-size: 0.85rem; color: var(--muted); margin-top: 0.35rem; line-height: 1.45; }

.card-icon.small {
    width: 60px; height: 60px; border-radius: 20px; display: grid; place-items: center; font-size: 1.7rem;
    background: linear-gradient(145deg, rgba(255,255,255,0.98), var(--tint));
    box-shadow: 0 12px 26px -12px var(--glow), inset 0 1px 0 #fff;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 999px; padding: 0.6rem 1.3rem; font-weight: 600;
    background: var(--glass-strong); color: var(--ink);
    border: 1px solid rgba(124,140,220,0.28);
    box-shadow: 0 6px 16px -10px rgba(80,90,190,0.35);
    transition: transform 0.25s var(--ease), box-shadow 0.25s var(--ease), border-color 0.25s, background 0.25s;
}
.stButton > button p { color: inherit; font-weight: 600; }
.stButton > button:hover {
    transform: translateY(-2px); background: #fff; border-color: rgba(124,147,255,0.7);
    box-shadow: 0 14px 28px -14px rgba(100,110,240,0.6); color: #3B45B8;
}
.stButton > button:active { transform: translateY(0) scale(0.99); }
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(115deg, #7C93FF 0%, #A08BFF 60%, #86D9C4 130%);
    border: none; color: #fff;
    box-shadow: 0 14px 30px -12px rgba(124,120,255,0.75);
}
.stButton > button[kind="primary"] p,
.stButton > button[data-testid="stBaseButton-primary"] p { color: #fff !important; }
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    color: #fff; background: linear-gradient(115deg, #6F87FF 0%, #9679FF 60%, #7ACFB8 130%);
    box-shadow: 0 20px 38px -14px rgba(124,120,255,0.85);
}
.stButton > button:focus-visible { outline: 3px solid rgba(124,147,255,0.55); outline-offset: 2px; }

/* ---------- Radio (problem picker) ---------- */
div[role="radiogroup"] { gap: 0.55rem; }
div[role="radiogroup"] > label {
    width: 100%; padding: 0.72rem 1rem; border-radius: 16px;
    background: var(--glass); border: 1px solid var(--edge);
    box-shadow: 0 4px 14px -10px rgba(80,90,190,0.35);
    transition: transform 0.25s var(--ease), border-color 0.25s, background 0.25s, box-shadow 0.25s;
}
div[role="radiogroup"] > label:hover { transform: translateX(3px); border-color: rgba(124,147,255,0.5); background: rgba(255,255,255,0.85); }
div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(120deg, rgba(143,168,255,0.24), rgba(195,180,255,0.28));
    border-color: rgba(124,147,255,0.65);
    box-shadow: 0 12px 26px -16px rgba(100,110,240,0.7);
}
div[role="radiogroup"] p { color: var(--ink); font-weight: 500; }

/* ---------- Text area ---------- */
.stTextArea label p { color: var(--ink); font-weight: 600; }
div[data-baseweb="textarea"], div[data-baseweb="base-input"] {
    border-radius: 18px !important; background: var(--glass-strong) !important;
    border: 1px solid rgba(124,140,220,0.25) !important;
    box-shadow: 0 6px 18px -12px rgba(80,90,190,0.35);
    transition: box-shadow 0.25s, border-color 0.25s;
}
div[data-baseweb="textarea"]:focus-within {
    border-color: rgba(124,147,255,0.8) !important;
    box-shadow: 0 0 0 4px rgba(143,168,255,0.22), 0 10px 26px -14px rgba(100,110,240,0.6);
}
.stTextArea textarea { background: transparent !important; color: var(--ink) !important; border-radius: 18px; }

/* ---------- Alerts, expanders, divider ---------- */
[data-testid="stAlert"] {
    border-radius: 18px; border: 1px solid var(--edge);
    background: var(--glass-strong); box-shadow: var(--shadow);
}
[data-testid="stExpander"] {
    border-radius: 18px; border: 1px solid var(--edge) !important;
    background: var(--glass); box-shadow: var(--shadow); overflow: hidden;
}
[data-testid="stExpander"] summary:hover { background: rgba(143,168,255,0.10); }
hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(124,147,255,0.4), transparent); margin: 1.8rem 0; }

/* ---------- Report card (bordered container) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 26px !important; border: 1px solid var(--edge) !important;
    background: var(--glass); backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    box-shadow: var(--shadow); padding: 0.6rem 0.8rem;
    animation: rise 0.55s var(--ease) both;
}
[data-testid="stVerticalBlockBorderWrapper"] h2 {
    font-size: 1.15rem; font-weight: 700; margin-top: 1.4rem; padding-bottom: 0.45rem;
    border-bottom: 1px solid transparent;
    border-image: linear-gradient(90deg, rgba(124,147,255,0.55), transparent) 1;
}
[data-testid="stVerticalBlockBorderWrapper"] h3 {
    font-size: 0.98rem; font-weight: 700; margin-top: 1rem; padding-left: 0.7rem;
    border-left: 3px solid; border-image: linear-gradient(180deg, #7C93FF, #86D9C4) 1;
}

/* ---------- Chips + status tags ---------- */
.chips { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
.chip {
    display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.3rem 0.85rem; border-radius: 999px;
    background: rgba(255,255,255,0.78); border: 1px solid rgba(120,130,200,0.2);
    font-size: 0.82rem; color: var(--muted);
}
.chip b { color: var(--ink); font-weight: 600; }
.source-tag-manual, .source-tag-ai, .conf {
    display: inline-block; padding: 0.3rem 0.85rem; border-radius: 999px; font-size: 0.8rem; font-weight: 700;
}
.source-tag-manual { background: #D7F7EA; color: #0F6B4A; }
.source-tag-ai { background: #FFEFCF; color: #8A5A00; }
.conf-high { background: #D7F7EA; color: #0F6B4A; }
.conf-medium { background: #FFEFCF; color: #8A5A00; }
.conf-low { background: #FFE0E3; color: #9C2B3A; }

.safety-box {
    margin-top: 1.1rem; padding: 1rem 1.2rem; border-radius: 18px; line-height: 1.5;
    background: linear-gradient(120deg, rgba(255,214,218,0.55), rgba(255,236,222,0.55));
    border: 1px solid rgba(255,150,160,0.5); color: #7A2530; font-size: 0.92rem;
}

/* ---------- Diagnosis output (quick view) ---------- */
.out-label {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
    color: var(--muted); margin: 1.7rem 0 0.65rem;
}
.quick-card {
    position: relative; overflow: hidden; border-radius: 28px; padding: 1.5rem 1.7rem 1.4rem;
    background: var(--glass); border: 1px solid var(--edge);
    backdrop-filter: blur(18px) saturate(140%); -webkit-backdrop-filter: blur(18px) saturate(140%);
    box-shadow: 0 2px 4px rgba(60,70,140,0.06), 0 26px 50px -22px var(--glow, rgba(120,130,255,0.5));
    animation: rise 0.55s var(--ease) both;
}
.quick-card::before {
    content: ""; position: absolute; width: 230px; height: 230px; border-radius: 50%;
    top: -100px; right: -80px; opacity: 0.65;
    background: radial-gradient(circle, var(--tint, #E4DAFF) 0%, transparent 70%);
}
.quick-card > * { position: relative; }
.card-icon.tiny {
    width: 44px; height: 44px; border-radius: 15px; display: grid; place-items: center; font-size: 1.25rem;
    background: linear-gradient(145deg, rgba(255,255,255,0.98), var(--tint, #E4DAFF));
    box-shadow: 0 10px 22px -10px var(--glow, rgba(120,130,255,0.5)), inset 0 1px 0 #fff;
}
.qc-top { display: flex; align-items: center; gap: 0.75rem; }
.qc-machine { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: var(--muted); }
.qc-problem { font-family: 'Bricolage Grotesque', sans-serif; font-size: clamp(1.5rem, 4vw, 2rem); font-weight: 700; letter-spacing: -0.02em; margin: 0.9rem 0 0.15rem; color: var(--ink); }
.qc-sub { color: var(--muted); font-size: 0.92rem; }
.qc-cause-label { color: var(--muted); font-size: 0.85rem; font-weight: 600; margin-top: 1.1rem; }
.qc-cause { font-size: 1.2rem; font-weight: 600; line-height: 1.4; color: var(--ink); margin-top: 0.15rem; }
.qc-meta { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.1rem; }

.check-list { display: grid; gap: 0.6rem; }
.check {
    display: flex; align-items: flex-start; gap: 0.9rem; padding: 0.85rem 1rem; border-radius: 18px;
    background: var(--glass); border: 1px solid var(--edge); box-shadow: 0 4px 14px -10px rgba(80,90,190,0.35);
    transition: transform 0.25s var(--ease), border-color 0.25s, box-shadow 0.25s;
}
.check:hover { transform: translateX(3px); border-color: rgba(124,147,255,0.5); box-shadow: 0 12px 26px -16px rgba(100,110,240,0.6); }
.check .num {
    flex-shrink: 0; width: 38px; height: 38px; border-radius: 12px; display: grid; place-items: center;
    font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 0.92rem; color: #3B45B8;
    background: linear-gradient(145deg, #fff, var(--tint, #E4DAFF));
    box-shadow: 0 8px 16px -10px var(--glow, rgba(120,130,255,0.6));
}
.check .t { font-weight: 600; color: var(--ink); line-height: 1.35; padding-top: 0.15rem; }
.check .d {
    font-size: 0.85rem; color: var(--muted); margin-top: 0.15rem; line-height: 1.45;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}

.next-card {
    padding: 1rem 1.2rem; border-radius: 20px; font-weight: 600; line-height: 1.5; color: var(--ink);
    background: linear-gradient(120deg, rgba(143,168,255,0.22), rgba(168,239,211,0.30));
    border: 1px solid rgba(124,147,255,0.4);
}
.safety-box .sb-title { font-weight: 700; font-size: 0.75rem; letter-spacing: 0.09em; text-transform: uppercase; margin-bottom: 0.4rem; }
[data-testid="stExpander"] h2 { font-size: 1.1rem; }
[data-testid="stExpander"] h3 { font-size: 1rem; }

/* ---------- Motion ---------- */
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) {
    .hero, .machine-head, .quick-card, [data-testid="stVerticalBlockBorderWrapper"] { animation: none; }
    .machine-card, .stButton > button, div[role="radiogroup"] > label, .machine-card .card-icon { transition: none; }
}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


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


def tint_style(machine):
    t = MACHINE_TINTS.get(machine, MACHINE_TINTS["Bearing"])
    return f"--tint:{t['tint']};--glow:{t['glow']};"


def extract_confidence(report_text):
    """Pull High / Medium / Low out of the '## Confidence' section, if present."""
    m = re.search(r"##\s*Confidence\s*\n+\W*(High|Medium|Low)", report_text or "", re.IGNORECASE)
    return m.group(1).capitalize() if m else None


def render_header(compact=False):
    """Hero on the home page, slim brand bar on the diagnose page."""
    if compact:
        st.markdown(
            '<div class="brand"><div class="logo">🛠️</div>'
            f'<div class="name">{html.escape(APP_TITLE)}</div></div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="hero">'
        '<div class="hero-ring"></div>'
        '<span class="pill"><span class="dot"></span>Manual-backed, AI-assisted</span>'
        f'<h1 class="hero-title">{html.escape(APP_TITLE)}</h1>'
        f'<p class="hero-sub">{html.escape(APP_TAGLINE)}</p>'
        '<div class="feature-row">'
        '<div class="feature" style="--tint:#D6E1FF;--glow:rgba(115,140,255,0.55);">'
        '<div class="f-icon">📖</div><div><b>Manual-backed</b><span>Checked against your knowledge base</span></div></div>'
        '<div class="feature" style="--tint:#E4DAFF;--glow:rgba(160,135,255,0.55);">'
        '<div class="f-icon">✨</div><div><b>AI reasoning</b><span>Fills gaps the manual misses</span></div></div>'
        '<div class="feature" style="--tint:#CFF5E6;--glow:rgba(90,205,160,0.55);">'
        '<div class="f-icon">🦺</div><div><b>Safety-first</b><span>Every report ends with safety notes</span></div></div>'
        '</div>'
        '<div class="notice"><b>Decision support, not a certified diagnosis.</b> '
        'It suggests possible causes and checks. For critical or unsafe conditions, call a qualified technician.</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def machine_card_html(machine):
    return (
        f'<div class="machine-card" style="{tint_style(machine)}">'
        f'<div class="card-icon">{MACHINE_ICONS[machine]}</div>'
        f'<div class="card-title">{html.escape(machine)}</div>'
        f'<div class="card-desc">{html.escape(MACHINE_DESCRIPTIONS[machine])}</div>'
        '</div>'
    )


# =========================================================================
# PAGES
# =========================================================================

def page_home(kb):
    render_header()
    st.markdown('<div class="section-title">Select a machine to begin</div>'
                '<div class="section-sub">Pick the equipment you are troubleshooting.</div>',
                unsafe_allow_html=True)

    cols = st.columns(3)
    machines = ["Pump", "Bearing", "Air Compressor"]
    for col, machine in zip(cols, machines):
        with col:
            st.markdown(machine_card_html(machine), unsafe_allow_html=True)
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
    render_header(compact=True)

    if st.button("← Back to machine selection"):
        go_to("home")
        st.rerun()

    st.markdown(
        f'<div class="machine-head"><div class="card-icon small" style="{tint_style(machine)}">'
        f'{MACHINE_ICONS.get(machine, "")}</div>'
        f'<div><div class="mh-title">{html.escape(str(machine))} diagnosis</div>'
        f'<div class="mh-desc">{html.escape(MACHINE_DESCRIPTIONS.get(machine, ""))}</div></div></div>',
        unsafe_allow_html=True,
    )

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


# =========================================================================
# DIAGNOSIS OUTPUT HELPERS
# These only READ the model's Markdown reply and reorganize how it is shown.
# The prompt, API call and diagnosis logic are not touched.
# =========================================================================

def _plain(text):
    """Strip Markdown emphasis and collapse whitespace (for short UI labels)."""
    text = re.sub(r"[*`]+", "", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _rich_html(text):
    """Escape text, keep **bold**, turn '-' / '*' bullets into '•', keep line breaks."""
    text = html.escape((text or "").strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    lines = [re.sub(r"^\s*[-*•]\s+", "• ", ln) for ln in text.splitlines() if ln.strip()]
    return "<br>".join(lines)


def split_report_sections(report_text):
    """Split the reply on '## ' headings -> {lowercase title: body}."""
    sections = {}
    parts = re.split(r"(?m)^##\s+(.+?)\s*$", report_text or "")
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i].strip().lower()] = parts[i + 1].strip()
    return sections


def find_section(sections, *keys):
    for title, body in sections.items():
        if any(k in title for k in keys):
            return body
    return ""


def parse_causes(body):
    """
    Parse the 'Possible Issues' section into a list of dicts:
    {tier, cause, why, source}. Returns [] if the format is not recognised.
    """
    items = []
    chunks = re.split(r"(?m)^###\s+(.+?)\s*$", body or "")
    blocks = [("Possible", chunks[0])]
    for i in range(1, len(chunks) - 1, 2):
        blocks.append((chunks[i].strip(), chunks[i + 1]))

    marker = re.compile(r"\*\*\s*Possible Cause\s*:?\s*\*\*\s*:?", re.I)
    for tier, text in blocks:
        for piece in marker.split(text)[1:]:
            cause = re.split(r"\*\*\s*(?:Why|Source)\s*:?\s*\*\*", piece, maxsplit=1, flags=re.I)[0]
            why_m = re.search(
                r"\*\*\s*Why\s*:?\s*\*\*\s*:?\s*(.*?)(?=\*\*\s*Source\s*:?\s*\*\*|\Z)",
                piece, flags=re.S | re.I,
            )
            src_m = re.search(r"\*\*\s*Source\s*:?\s*\*\*\s*:?\s*(.*)", piece, flags=re.S | re.I)
            source = ""
            if src_m and src_m.group(1).strip():
                source = _plain(src_m.group(1).strip().splitlines()[0])
            cause = _plain(cause)
            if not cause:
                continue
            items.append({
                "tier": tier,
                "cause": cause,
                "why": why_m.group(1).strip() if why_m else "",
                "source": source,
            })
    return items


def parse_steps(body):
    """Return the list of numbered/bulleted troubleshooting steps."""
    return re.findall(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+(.+?)\s*$", body or "")


def split_check_text(text, limit=70):
    """Turn a full step sentence into (short title, optional detail)."""
    text = _plain(text)
    parts = re.split(r"(?<=[.:;])\s+|\s[-\u2013\u2014]\s", text, maxsplit=1)
    title = parts[0].rstrip(".:; ")
    rest = parts[1].strip() if len(parts) > 1 else ""
    if len(title) > limit:
        cut = title[:limit].rsplit(" ", 1)[0].rstrip(",;:- ")
        title, rest = cut + "…", text
    return title, rest


def _is_manual(item):
    return "manual" in item["source"].lower()


def _is_ai(item):
    return bool(re.search(r"\bai\b", item["source"].lower()))


def _render_items(items):
    for it in items:
        st.markdown(f"**{it['cause']}** ({it['tier']})")
        if it["why"]:
            st.markdown(it["why"])
        if it["source"]:
            st.caption(f"Source: {it['source']}")


# =========================================================================
# DIAGNOSIS OUTPUT (Quick diagnosis -> Checks -> Next check -> Safety -> Details)
# =========================================================================

def render_report(machine, problem, matched_entry, report_text):
    st.divider()

    sections = split_report_sections(report_text)
    summary = find_section(sections, "summary")
    causes = parse_causes(find_section(sections, "possible issues", "possible causes"))
    troubleshooting_body = find_section(sections, "troubleshooting")
    steps = parse_steps(troubleshooting_body)
    next_check = _plain(find_section(sections, "next check", "next"))
    confidence = extract_confidence(report_text)
    conf_body = find_section(sections, "confidence")
    conf_reason = ""
    m = re.search(r"(?:High|Medium|Low)\W*(.*)", conf_body, flags=re.S | re.I)
    if m:
        conf_reason = _plain(m.group(1))
    safety_body = find_section(sections, "safety")

    top = next((c for c in causes if "most" in c["tier"].lower()), causes[0] if causes else None)
    other_causes = [c for c in causes if c is not top]
    manual_items = [c for c in causes if _is_manual(c)]
    ai_items = [c for c in causes if _is_ai(c)]

    parsed_ok = bool(top or steps or next_check)

    # ---- 1. QUICK DIAGNOSIS ------------------------------------------------
    if problem == "Other":
        headline = "Reported issue"
        desc = _plain(st.session_state.get("description", ""))
        sub = desc if len(desc) <= 140 else desc[:140].rsplit(" ", 1)[0] + "…"
    else:
        headline, sub = str(problem), ""

    if top:
        cause_text = top["cause"]
    elif summary:
        first = re.split(r"(?<=[.!?])\s", _plain(summary), maxsplit=1)[0]
        cause_text = first
    else:
        cause_text = "See the full report below."

    source_html = (
        '<span class="source-tag-manual">Manual-supported problem</span>'
        if matched_entry
        else '<span class="source-tag-ai">Primarily AI reasoning</span>'
    )
    conf_html = (
        f'<span class="conf conf-{confidence.lower()}">Confidence: {confidence}</span>'
        if confidence else ""
    )

    st.markdown('<div class="out-label">Quick diagnosis</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="quick-card" style="{tint_style(machine)}">'
        '<div class="qc-top">'
        f'<div class="card-icon tiny">{MACHINE_ICONS.get(machine, "🛠️")}</div>'
        f'<div class="qc-machine">{html.escape(str(machine))} diagnosis</div></div>'
        f'<div class="qc-problem">⚠️ {html.escape(headline)}</div>'
        + (f'<div class="qc-sub">{html.escape(sub)}</div>' if sub else "")
        + '<div class="qc-cause-label">Most likely cause</div>'
        f'<div class="qc-cause">{html.escape(cause_text)}</div>'
        f'<div class="qc-meta">{conf_html}{source_html}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- 2. TOP RECOMMENDED CHECKS ----------------------------------------
    if steps:
        rows = []
        for i, step in enumerate(steps[:3], start=1):
            title, rest = split_check_text(step)
            rows.append(
                '<div class="check">'
                f'<div class="num">{i:02d}</div>'
                '<div class="ct">'
                f'<div class="t">{html.escape(title)}</div>'
                + (f'<div class="d">{html.escape(rest)}</div>' if rest else "")
                + '</div></div>'
            )
        st.markdown('<div class="out-label">What to check</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="check-list" style="{tint_style(machine)}">{"".join(rows)}</div>',
            unsafe_allow_html=True,
        )

    # ---- 3. NEXT RECOMMENDED CHECK ----------------------------------------
    if next_check:
        st.markdown('<div class="out-label">Next recommended check</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="next-card">{html.escape(next_check)}</div>', unsafe_allow_html=True)

    # ---- 4. SAFETY WARNING -------------------------------------------------
    model_safety = _rich_html(safety_body)
    st.markdown(
        '<div class="safety-box"><div class="sb-title">⚠️ Safety warning</div>'
        + (f'{model_safety}<br><br>' if model_safety else "")
        + 'Always shut down and isolate the machine safely before physical inspection. '
        'If the problem persists or requires specialist repair, contact a qualified technician.</div>',
        unsafe_allow_html=True,
    )

    # ---- 5. DETAILED REPORT (collapsed by default) ------------------------
    st.markdown('<div class="out-label">Detailed report</div>', unsafe_allow_html=True)

    if summary or (top and top["why"]):
        with st.expander("Why this diagnosis is likely"):
            if summary:
                st.markdown(summary)
            if top and top["why"]:
                st.markdown(f"**Main reasoning ({top['cause']}):** {top['why']}")

    if other_causes:
        with st.expander("Other possible causes"):
            _render_items(other_causes)

    if troubleshooting_body:
        with st.expander("Detailed troubleshooting"):
            st.markdown(troubleshooting_body)

    if causes or conf_reason:
        with st.expander("Evidence / reasoning"):
            _render_items(causes)
            if conf_reason:
                st.markdown(f"**Confidence ({confidence or 'n/a'}):** {conf_reason}")

    if manual_items:
        with st.expander("Manual-based findings"):
            _render_items(manual_items)

    if ai_items:
        with st.expander("AI-based possibilities"):
            _render_items(ai_items)

    if matched_entry:
        with st.expander("Matched manual excerpt"):
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

    # Always keep the complete, unmodified AI reply available.
    with st.expander("Full AI report (original text)", expanded=not parsed_ok):
        st.markdown(report_text)


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
