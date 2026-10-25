
import os
import re
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq, GroqError
from extractor import download_filings

st.set_page_config(page_title="SECInsight", page_icon="📊", layout="wide")
load_dotenv()

try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    api_key = None
api_key = api_key or os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None
MODEL_ID = "openai/gpt-oss-120b"

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1.6rem;}
h1 {color: #173b66; letter-spacing: -1px;}
.small-note {color: #64748b; font-size: .92rem;}
div[data-testid="stMetric"] {border: 1px solid #e2e8f0; padding: 14px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)
st.title("SECInsight")
st.caption("Evidence-based analysis of SEC 10-K filings and year-over-year financial changes")

def run_groq(prompt, max_tokens=1000):
    if client is None:
        raise RuntimeError("GROQ_API_KEY is missing from Streamlit Secrets or environment variables.")
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": (
                    "You are a careful SEC filing analyst. Never invent financial figures. "
                    "Separate reported facts from interpretation. If evidence is missing, say so."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        result = response.choices[0].message.content
        if not result or not result.strip():
            raise RuntimeError("Groq returned an empty response.")
        return result.strip()
    except GroqError as exc:
        status = getattr(exc, "status_code", None)
        messages = {
            401: "Groq API key is invalid.",
            404: f"Groq model {MODEL_ID} was not found.",
            413: "Request too large. Reduce the extracted filing sections.",
            429: "Groq rate limit or usage quota reached.",
        }
        raise RuntimeError(messages.get(status, f"Groq API error: HTTP {status or 'unknown'}.")) from exc


def analyze_with_ai(filing):
    content = filing.get("analysis_content", "").strip()
    financials = filing.get("financials", {})
    facts = {
        key: financials.get(key)
        for key in ("Revenue", "Net income", "Operating cash flow")
    }
    if len(content) < 500:
        return {
            "error": "The actual 10-K sections could not be extracted reliably for this filing.",
            "raw": "",
            "data": None,
        }

    prompt = f"""
Analyze the SEC 10-K for {filing['ticker']}, fiscal year {filing['year']}.
Use only the evidence below. Never invent facts or numbers.
Financial figures below come from SEC XBRL CompanyFacts, when available.
Filing excerpt:
{content[:10000]}

Structured financial facts (raw USD values where available):
{json.dumps(facts)}

Return valid JSON only, with exactly these keys:
performance_summary (string),
top_risks (array of up to 3 specific risks),
management_tone (Optimistic, Neutral, Cautious, or Insufficient Data),
investment_signal (Positive, Neutral, Negative, or Insufficient Data),
signal_evidence (string),
evidence_limitations (string).

