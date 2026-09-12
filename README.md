# 🛠️ AI Machine Diagnosis Assistant

A Streamlit app that helps users troubleshoot common problems in **Pumps**, **Bearings**,
and **Air Compressors**. It uses a hybrid approach: a built-in technical knowledge base
(`knowledge_base/machine_manual.json`) plus an LLM (via the free-tier [Groq API](https://console.groq.com))
for general mechanical reasoning when the knowledge base doesn't cover the reported problem.

> ⚠️ This is a **decision-support tool**, not a certified diagnosis. Always follow safe
> shutdown/isolation procedures and consult a qualified technician for anything beyond
> basic inspection.

---

## Project Structure

```text
machine-diagnosis/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
└── knowledge_base/
    └── machine_manual.json         # Structured troubleshooting knowledge base
```

`machine_manual.json` is used instead of a raw PDF so the app can look up relevant
entries instantly without needing a PDF-parsing/embedding pipeline — you can freely edit
or extend this file (add problems, causes, steps) without touching `app.py`.

---

## 1. Get the project onto your machine

Create a folder named `machine-diagnosis` and place `app.py`, `requirements.txt`,
`README.md`, and the `knowledge_base/machine_manual.json` file inside it exactly as
shown in the structure above.

## 2. Install dependencies

From inside the `machine-diagnosis` folder:

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Get a free Groq API key

1. Sign up at [console.groq.com](https://console.groq.com).
2. Create an API key from the dashboard.
3. Keep it handy — you'll add it to Streamlit secrets next (never paste it into `app.py`).

## 4. Configure the API key locally

Create a file at `.streamlit/secrets.toml` (create the `.streamlit` folder if it doesn't
exist) inside your project folder:

```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

**Do not commit this file to GitHub.** Add `.streamlit/secrets.toml` to a `.gitignore` file.

## 5. Run the app locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. It will start even without
an API key — you'll simply see a friendly error message if you try to run a diagnosis
before adding one.

## 6. Upload the project to GitHub

```bash
git init
git add app.py requirements.txt README.md knowledge_base/machine_manual.json .gitignore
git commit -m "Initial commit: AI Machine Diagnosis Assistant"
git branch -M main
git remote add origin https://github.com/<your-username>/machine-diagnosis.git
git push -u origin main
```

(Make sure `.streamlit/secrets.toml` is **not** included — it should stay only on your
local machine.)

## 7. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app** and select your `machine-diagnosis` repository and the `main` branch.
3. Set the main file path to `app.py`.
4. Before (or after) deploying, open **Settings → Secrets** for the app and paste:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
5. Click **Deploy**. Streamlit Cloud will install `requirements.txt` automatically.

---

## How it works

1. **Select a machine** (Pump, Bearing, or Air Compressor) from the home screen.
2. **Select a common problem** from the list, or choose **Other** and describe your
   symptoms in your own words.
3. The app performs a lightweight keyword match against `machine_manual.json` to see if
   the problem is covered by the knowledge base.
   - If a match is found, the relevant manual excerpt is sent to the LLM as context, and
     findings are labeled **Manual-Based Finding**.
   - If no relevant match is found, the LLM reasons from general mechanical engineering
     knowledge, and findings are labeled **AI-Based Possibility**.
4. If the description is too short for a reliable assessment (only relevant to the
   "Other" path), the app asks a few targeted follow-up questions instead of guessing.
5. The app calls the Groq API **once per diagnosis** (no repeated/duplicate calls) and
   displays a structured report: diagnosis summary, ranked possible causes, suggested
   troubleshooting steps, a recommended next check, a confidence level, and a safety
   warning.

## Changing the AI model

The model name is centralized in one place in `app.py`:

```python
MODEL_NAME = "llama-3.3-70b-versatile"
```

If Groq retires this model, check the current list at
[console.groq.com/docs/models](https://console.groq.com/docs/models) and update this
single line — no other code changes are needed.

## Extending to new machines or problems

- To add a new common problem to an existing machine, add an entry to that machine's
  `"common_problems"` list and a matching object in `"entries"` in
  `knowledge_base/machine_manual.json` — no Python changes required.
- To add a whole new machine type, add a new top-level key to `machine_manual.json`
  following the same structure, and add it to the `machines` list, `MACHINE_ICONS`, and
  `MACHINE_DESCRIPTIONS` in `app.py`.

## What was validated

Because code changes here can't be executed inside your exact deployment environment,
here's exactly what was checked before delivery:

- ✅ `python -m py_compile app.py` — no syntax errors.
- ✅ `knowledge_base/machine_manual.json` parses as valid JSON; every machine has
  `common_problems` and matching `entries`.
- ✅ The app was started locally with `streamlit run app.py` (headless) and came up
  cleanly with no missing-import or undefined-name errors, including **without** a
  `GROQ_API_KEY` set (confirms the app doesn't crash when the key is missing).
- ✅ Manually traced all five required test scenarios (pump vibration, bearing noise,
  compressor low pressure, pump "Other" with a described stop-after-minutes symptom,
  and an insufficiently-described "Other" case) through the code logic:
  `retrieve_manual_match`, `is_input_sufficient`, `run_diagnosis`, and `render_report`.
- ✅ Every function referenced in the UI flow is defined; every dependency imported is
  listed in `requirements.txt`.

**Recommended local test procedure before your demo:**
1. Run `streamlit run app.py` with a valid `GROQ_API_KEY` set.
2. Pump → "Excessive vibration" → Run Diagnosis → confirm a manual-labeled report appears.
3. Bearing → "Unusual noise" → Run Diagnosis → confirm relevant causes and no false certainty.
4. Air Compressor → "Low pressure" → Run Diagnosis → confirm causes + steps appear.
5. Pump → "Other" → type "The pump starts normally but stops after running for several
   minutes." → Run Diagnosis → confirm it's accepted and clearly labeled (manual or AI).
6. Pump → "Other" → type a very short phrase (1-2 words) → confirm follow-up questions
   are shown instead of a guessed diagnosis.
7. Temporarily remove `GROQ_API_KEY` from secrets and confirm the app shows a friendly
   error instead of crashing.
