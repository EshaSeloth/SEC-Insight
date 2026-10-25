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

Use Insufficient Data if evidence is inadequate. Do not label a year Negative merely because risks are disclosed.
A qualitative signal is not a stock-price prediction or investment recommendation.
"""
    try:
        raw = run_groq(prompt)
    except Exception as exc:
        return {"error": str(exc), "raw": "", "data": None}

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        return {"error": "AI response was not valid JSON.", "raw": raw, "data": None}
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return {"error": "AI returned malformed JSON.", "raw": raw, "data": None}

    tones = {"Optimistic", "Neutral", "Cautious", "Insufficient Data"}
    signals = {"Positive", "Neutral", "Negative", "Insufficient Data"}
    tone = str(data.get("management_tone", "Insufficient Data")).strip().title()
    signal = str(data.get("investment_signal", "Insufficient Data")).strip().title()
    data["management_tone"] = tone if tone in tones else "Insufficient Data"
    data["investment_signal"] = signal if signal in signals else "Insufficient Data"
    data["performance_summary"] = str(data.get("performance_summary", "Insufficient Data"))
    risks = data.get("top_risks", [])
    data["top_risks"] = risks[:3] if isinstance(risks, list) else []
    data["signal_evidence"] = str(data.get("signal_evidence", ""))
    data["evidence_limitations"] = str(data.get("evidence_limitations", ""))
    return {"error": None, "raw": raw, "data": data}


def compare_years(results):
    summaries = [{
        "year": r["year"],
        "financials": r.get("financials", {}),
        "performance_summary": r["analysis"].get("performance_summary", ""),
        "top_risks": r["analysis"].get("top_risks", []),
        "management_tone": r["management_tone"],
        "investment_signal": r["sentiment"],
        "signal_evidence": r["analysis"].get("signal_evidence", ""),
    } for r in results]
    return run_groq(
        "Compare these SEC filing analyses chronologically. Use only supplied evidence. "
        "Call out missing/incomparable data. Do not invent trends. Keep concise.\n"
        + json.dumps(summaries, ensure_ascii=False),
        max_tokens=700,
    )


def answer_question(question, results):
    context = [{
        "year": r["year"],
        "financials": r.get("financials", {}),
        "analysis": r["analysis"],
        "source_excerpt": r.get("source_excerpt", "")[:1200],
    } for r in results]
    return run_groq(
        "Answer using only these filing analyses and source excerpts. Do not invent facts. "
        "Name fiscal years when possible; state when evidence is absent.\n"
        f"Question: {question}\nEvidence: {json.dumps(context, ensure_ascii=False)}",
        max_tokens=700,
    )


st.divider()
left_space, center_panel, right_space = st.columns([1, 2, 1])
with center_panel:
    st.subheader("Analyze filings")
    ticker_input = st.text_input("Stock ticker", value="AAPL", placeholder="AAPL, MSFT, TSLA")
    years = st.selectbox("Number of fiscal years", options=[1, 2, 3, 4, 5], index=2)
    run_button = st.button("Analyze filings", type="primary", use_container_width=True)
    st.caption("Financial charts use SEC-reported XBRL facts when available.")
st.divider()

if "sec_results" not in st.session_state:
    st.session_state.sec_results = None
if "sec_comparison" not in st.session_state:
    st.session_state.sec_comparison = None
if "sec_answer" not in st.session_state:
    st.session_state.sec_answer = None
if "sec_ticker" not in st.session_state:
    st.session_state.sec_ticker = None

if run_button:
    ticker = ticker_input.strip().upper()
    st.session_state.sec_results = None
    st.session_state.sec_comparison = None
    st.session_state.sec_answer = None
    if not ticker:
        st.error("Enter a ticker symbol.")
    elif client is None:
        st.error("Add GROQ_API_KEY to Streamlit Cloud Secrets.")
    else:
        try:
            with st.status("Retrieving SEC filings and financial facts...", expanded=True) as status:
                filings = download_filings(ticker, years)
                status.write(f"Retrieved {len(filings)} annual filing(s).")
                status.update(label="SEC retrieval complete", state="complete")

            results = []
            for filing in sorted(filings, key=lambda f: int(f["year"])):
                filing["ticker"] = ticker
                with st.spinner(f"Analyzing fiscal year {filing['year']}..."):
                    output = analyze_with_ai(filing)
                if output["error"]:
                    st.warning(f"{filing['year']}: {output['error']}")
                    with st.expander(f"Extraction details: {filing['year']}"):
                        st.write("Extracted text length:", len(filing.get("analysis_content", "")))
                        st.text(filing.get("analysis_content", "")[:2500] or "No section text extracted.")
                        if output["raw"]:
                            st.text(output["raw"][:3000])
                    continue
                data = output["data"]
                results.append({
                    "year": int(filing["year"]),
                    "filed_date": filing.get("filed_date", ""),
                    "sentiment": data["investment_signal"],
                    "management_tone": data["management_tone"],
                    "analysis": data,
                    "source_excerpt": filing.get("analysis_content", ""),
                    "financials": filing.get("financials", {}),
                    "financials_error": filing.get("financials_error", ""),
                })

            if not results:
                st.error("No filing years could be analyzed. Check extraction details above.")
            else:
                st.session_state.sec_results = results
                st.session_state.sec_ticker = ticker
                try:
                    with st.spinner("Comparing years..."):
                        st.session_state.sec_comparison = compare_years(results)
                except Exception as exc:
                    st.session_state.sec_comparison = f"Comparison unavailable: {exc}"
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            with st.expander("Technical details"):
                st.exception(exc)

if st.session_state.sec_results:
    results = st.session_state.sec_results
    ticker = st.session_state.sec_ticker
    latest = results[-1]
    st.success(f"Analysis complete for {ticker}")
    a, b, c = st.columns(3)
    a.metric("Ticker", ticker)
    b.metric("Years analyzed", len(results))
    c.metric("Latest qualitative signal", latest["sentiment"])
    st.caption("Qualitative signals are interpretations of filing evidence, not investment advice or stock-return predictions.")

    st.subheader("Financial trends")
    metric_choice = st.selectbox("Metric", ["Revenue", "Net income", "Operating cash flow"])
    financial_rows = []
    for r in results:
        value = (r.get("financials") or {}).get(metric_choice)
        if isinstance(value, (int, float)):
            financial_rows.append({"Fiscal year": r["year"], metric_choice: value / 1_000_000_000})
    if financial_rows:
        fin_df = pd.DataFrame(financial_rows).drop_duplicates("Fiscal year").sort_values("Fiscal year")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(fin_df["Fiscal year"], fin_df[metric_choice], marker="o", linewidth=2)
        ax.set_title(f"{ticker}: {metric_choice} by fiscal year")
        ax.set_xlabel("Fiscal year")
        ax.set_ylabel(f"{metric_choice} (USD billions)")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.dataframe(fin_df.rename(columns={metric_choice: f"{metric_choice} (USD billions)"}), use_container_width=True, hide_index=True)
    else:
        st.info("SEC CompanyFacts did not return this metric for the selected years. Try another metric or ticker.")

    st.subheader("Yearly analysis")
    for result in results:
        data = result["analysis"]
        with st.expander(f"{result['year']} · {result['sentiment']}", expanded=True):
            st.write("**Management tone**", result["management_tone"])
            st.write("**Performance summary**", data["performance_summary"])
            st.write("**Top risks**")
            if data["top_risks"]:
                for risk in data["top_risks"]:
                    st.write(f"- {risk}")
            else:
                st.write("No specific risks extracted.")
            st.write("**Evidence for signal**", data["signal_evidence"] or "Not provided.")
            st.write("**Evidence limitations**", data["evidence_limitations"] or "Not provided.")
            with st.expander("Inspect source text"):
                st.text(result["source_excerpt"][:3000] or "No section text available.")
            if result.get("financials_error"):
                st.caption(f"Financial facts lookup issue: {result['financials_error']}")

    st.subheader("Year-over-year comparison")
    st.write(st.session_state.sec_comparison or "Comparison unavailable.")

    st.subheader("Ask about these filings")
    with st.form("sec_question_form"):
        question = st.text_input("Question", placeholder="What changed in revenue or risk disclosures?")
        ask = st.form_submit_button("Ask")
    if ask:
        if not question.strip():
            st.warning("Enter a question.")
        else:
            try:
                st.session_state.sec_answer = answer_question(question.strip(), results)
            except Exception as exc:
                st.session_state.sec_answer = f"Answer unavailable: {exc}"
    if st.session_state.sec_answer:
        st.info(st.session_state.sec_answer)

    st.subheader("Results table")
    rows = []
    for r in results:
        facts = r.get("financials") or {}
        rows.append({
            "Fiscal year": r["year"],
            "Filing date": r["filed_date"],
            "Revenue (USD)": facts.get("Revenue"),
            "Net income (USD)": facts.get("Net income"),
            "Operating cash flow (USD)": facts.get("Operating cash flow"),
            "Management tone": r["management_tone"],
            "Qualitative signal": r["sentiment"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{ticker}_secinsight.csv", mime="text/csv")

