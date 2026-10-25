
import os
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq, GroqError
from extractor import download_filings

st.set_page_config(page_title="SECInsight", layout="wide")
load_dotenv()

try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    api_key = None

api_key = api_key or os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None
MODEL_ID = "openai/gpt-oss-120b"

def run_groq(prompt, max_tokens=1000):
    if client is None:
        raise RuntimeError("GROQ_API_KEY is missing from Streamlit Secrets or environment variables.")
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful financial analyst. Never invent financial data. Clearly distinguish reported facts from interpretation."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens
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
            429: "Groq rate limit or usage quota reached."
        }
        raise RuntimeError(messages.get(status, f"Groq API error: HTTP {status or 'unknown'}.")) from exc

def analyze_with_ai(filing):
    content = filing.get("analysis_content", "").strip()
    if len(content) < 500:
        return {
            "error": "The extractor could not find enough text in the actual 10-K business, risk, or management discussion sections.",
            "raw": "",
            "data": None
        }

    prompt = f"""
Analyze this SEC 10-K excerpt for {filing['ticker']}, fiscal year {filing['year']}.
Use only the excerpt. Do not invent figures, growth rates, causes, or facts.
Do not infer stock-price performance from management language.
Return valid JSON only, with exactly these keys:
performance_summary: string
top_risks: array of up to 3 specific risks
management_tone: one of Optimistic, Neutral, Cautious, Insufficient Data
investment_signal: one of Positive, Neutral, Negative, Insufficient Data
signal_evidence: string
evidence_limitations: string

Choose Insufficient Data if the excerpt does not support a conclusion. Positive means the supplied evidence shows more favorable than unfavorable business signals; Negative means more unfavorable than favorable signals; Neutral means mixed or broadly stable evidence. These are qualitative interpretations, not investment recommendations.

Filing excerpt:
{content[:9000]}
"""
    try:
        raw = run_groq(prompt)
    except Exception as exc:
        return {"error": str(exc), "raw": "", "data": None}

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        return {
            "error": "The AI response was not valid JSON. Inspect the raw response below.",
            "raw": raw,
            "data": None
        }
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return {
            "error": "The AI returned malformed JSON. Inspect the raw response below.",
            "raw": raw,
            "data": None
        }

    allowed_tones = {"Optimistic", "Neutral", "Cautious", "Insufficient Data"}
    allowed_signals = {"Positive", "Neutral", "Negative", "Insufficient Data"}
    tone = str(data.get("management_tone", "Insufficient Data")).strip().title()
    signal = str(data.get("investment_signal", "Insufficient Data")).strip().title()

    if tone not in allowed_tones:
        tone = "Insufficient Data"
    if signal not in allowed_signals:
        signal = "Insufficient Data"

    data["management_tone"] = tone
    data["investment_signal"] = signal
    data["performance_summary"] = str(data.get("performance_summary", "Insufficient Data"))
    data["top_risks"] = data.get("top_risks", [])
    data["signal_evidence"] = str(data.get("signal_evidence", ""))
    data["evidence_limitations"] = str(data.get("evidence_limitations", ""))
    return {"error": None, "raw": raw, "data": data}

def compare_years(results):
    summaries = [
        {
            "year": r["year"],
            "performance_summary": r["analysis"].get("performance_summary", ""),
            "top_risks": r["analysis"].get("top_risks", []),
            "management_tone": r["management_tone"],
            "investment_signal": r["sentiment"],
            "signal_evidence": r["analysis"].get("signal_evidence", "")
        }
        for r in results
    ]
    return run_groq(
        "Compare these year-labelled SEC filing analyses chronologically. "
        "Use only the provided information. Do not invent trends. "
        "Call out missing or incomparable evidence. Keep the answer concise.\n\n"
        + json.dumps(summaries, ensure_ascii=False),
        max_tokens=700
    )

def answer_question(question, results):
    context = []
    for r in results:
        context.append({
            "year": r["year"],
            "analysis": r["analysis"],
            "source_excerpt": r.get("source_excerpt", "")[:1200]
        })
    return run_groq(
        f"Answer using only these SEC filing excerpts and analyses. "
        f"Do not invent facts. Identify the fiscal year when possible. "
        f"If evidence is absent, say so.\nQuestion: {question}\n"
        f"Evidence: {json.dumps(context, ensure_ascii=False)}",
        max_tokens=700
    )

st.markdown("""
<style>
.block-container{max-width:1100px;padding-top:2rem}
.header{text-align:center;padding:1rem 0}
.header h1{font-size:2.7rem;font-weight:800;color:#1a3f6f;margin:0}
.header h1 span{color:#c9a84c}
.header p{color:#64748b}
</style>
<div class="header">
<h1>SEC<span>Insight</span></h1>
<p>Evidence-based analysis of SEC 10-K filings and year-over-year changes</p>
</div>
""", unsafe_allow_html=True)

st.divider()

ticker = st.text_input("Stock ticker", value="AAPL", placeholder="AAPL, MSFT, TSLA").strip().upper()
years = st.slider("Number of fiscal years", 1, 5, 3)
run_button = st.button("Analyze filings", type="primary", use_container_width=True)

if "sec_results" not in st.session_state:
    st.session_state.sec_results = None
if "sec_comparison" not in st.session_state:
    st.session_state.sec_comparison = None
if "sec_ticker" not in st.session_state:
    st.session_state.sec_ticker = None
if "sec_answer" not in st.session_state:
    st.session_state.sec_answer = None

if run_button:
    st.session_state.sec_results = None
    st.session_state.sec_comparison = None
    st.session_state.sec_answer = None

    if not ticker:
        st.error("Enter a ticker symbol.")
    elif client is None:
        st.error("Add GROQ_API_KEY to Streamlit Cloud Secrets.")
    else:
        try:
            with st.status("Downloading SEC filings...", expanded=True) as status:
                filings = download_filings(ticker, years)
                status.write(f"Found {len(filings)} filing records.")
                status.update(label="Filing retrieval finished", state="complete")

            filings = sorted(filings, key=lambda f: int(f["year"]))[-years:]
            results = []

            for filing in filings:
                filing["ticker"] = ticker
                with st.spinner(f"Analyzing fiscal year {filing['year']}..."):
                    output = analyze_with_ai(filing)

                if output["error"]:
                    st.error(f"Fiscal year {filing['year']}: {output['error']}")
                    with st.expander(f"Debug details: {filing['year']}"):
                        st.write("Extracted section text length:", len(filing.get("analysis_content", "")))
                        st.text(filing.get("analysis_content", "")[:2500])
                        if output["raw"]:
                            st.write("Raw AI response")
                            st.text(output["raw"][:4000])
                    continue

                data = output["data"]
                results.append({
                    "year": int(filing["year"]),
                    "filed_date": filing.get("filed_date", ""),
                    "sentiment": data["investment_signal"],
                    "management_tone": data["management_tone"],
                    "analysis": data,
                    "source_excerpt": filing.get("analysis_content", "")
                })

            if not results:
                st.error("No years were analyzed successfully. Open the debug details above to see whether extraction or AI output failed.")
            else:
                st.session_state.sec_results = sorted(results, key=lambda r: r["year"])
                st.session_state.sec_ticker = ticker
                with st.spinner("Comparing analyzed years..."):
                    try:
                        st.session_state.sec_comparison = compare_years(st.session_state.sec_results)
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
    c.metric("Latest signal", latest["sentiment"])

    st.caption("Signals are qualitative interpretations of selected filing sections, not investment advice or stock-return predictions.")

    st.subheader("Yearly analysis")
    for result in results:
        data = result["analysis"]
        with st.expander(f"{result['year']} | {result['sentiment']}", expanded=True):
            st.write("**Management tone:**", result["management_tone"])
            st.write("**Performance summary**")
            st.write(data["performance_summary"])
            st.write("**Top risks**")
            risks = data["top_risks"]
            if isinstance(risks, list) and risks:
                for risk in risks[:3]:
                    st.write(f"- {risk}")
            else:
                st.write("No specific risks were returned.")
            st.write("**Evidence for signal**")
            st.write(data["signal_evidence"] or "Not provided.")
            st.write("**Evidence limitations**")
            st.write(data["evidence_limitations"] or "Not provided.")
            with st.expander("Inspect source text"):
                st.text(result["source_excerpt"][:4000])

    st.subheader("Year-over-year comparison")
    st.write(st.session_state.sec_comparison or "Comparison unavailable.")

    st.subheader("Ask about these filings")
    with st.form("sec_question_form"):
        question = st.text_input("Question", placeholder="What risks changed across these years?")
        ask = st.form_submit_button("Ask")
    if ask:
        if not question.strip():
            st.warning("Enter a question.")
        else:
            try:
                with st.spinner("Checking the supplied evidence..."):
                    st.session_state.sec_answer = answer_question(question.strip(), results)
            except Exception as exc:
                st.session_state.sec_answer = f"Answer unavailable: {exc}"
    if st.session_state.sec_answer:
        st.info(st.session_state.sec_answer)

    st.subheader("Results table")
    df = pd.DataFrame([{
        "Fiscal Year": r["year"],
        "Filing Date": r["filed_date"],
        "Management Tone": r["management_tone"],
        "Investment Signal": r["sentiment"]
    } for r in results])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False),
        file_name=f"{ticker}_secinsight.csv",
        mime="text/csv"
    )

    st.subheader("Sentiment trend")
    scores = {"Negative": -1, "Neutral": 0, "Positive": 1}
    plot_df = pd.DataFrame([
        {"year": r["year"], "score": scores[r["sentiment"]]}
        for r in results if r["sentiment"] in scores
    ])
    if plot_df.empty:
        st.warning("No valid sentiment labels are available, so no graph is shown.")
    else:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(plot_df["year"], plot_df["score"], marker="o", linewidth=2)
        ax.set_yticks([-1, 0, 1], ["Negative", "Neutral", "Positive"])
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("Fiscal year")
        ax.set_title(f"{ticker} filing sentiment")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
