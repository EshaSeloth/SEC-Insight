
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq, GroqError
from extractor import download_filings

st.set_page_config(page_title="SECInsight", layout="centered")
load_dotenv()

try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    api_key = None

api_key = api_key or os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None
MODEL_ID = "openai/gpt-oss-120b"

def run_groq(prompt):
    if client is None:
        st.error("Missing GROQ_API_KEY. Add it to Streamlit Secrets.")
        return None
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=900,
        )
        return response.choices[0].message.content
    except GroqError as exc:
        status = getattr(exc, "status_code", None)
        messages = {
            401: "Invalid Groq API key.",
            404: f"Model '{MODEL_ID}' was not found.",
            413: "The request is too large. Reduce the filing excerpt size.",
            429: "Groq usage or rate limit reached.",
        }
        st.error(messages.get(status, f"Groq request failed (HTTP {status or 'unknown'})."))
        return None
    except Exception as exc:
        st.error(f"Unexpected AI error: {exc}")
        return None

def analyze_with_ai(filing):
    content = filing.get("analysis_content") or filing.get("content", "")
    content = content[:8000]

    if len(content.strip()) < 1000:
        return "Insufficient filing text: the extracted content is too short to analyze reliably."

    prompt = f"""
You are a careful financial analyst reviewing an SEC 10-K.
Ticker: {filing.get('ticker', 'Unknown')}
Fiscal year: {filing['year']}

Use only evidence present in the filing excerpt. Do not invent facts.

Return these exact headings:

PERFORMANCE SUMMARY:
Summarize reported performance in two concise sentences.

TOP THREE RISKS:
List up to three specific supported risks. State if evidence is insufficient.

MANAGEMENT TONE:
Start with exactly one label: Optimistic, Neutral, Cautious, or Insufficient Data. Then briefly explain.

INVESTMENT SIGNAL:
Start with exactly one label: Positive, Neutral, Negative, or Insufficient Data. Assess only the supplied evidence, not expected stock returns.

EVIDENCE LIMITATIONS:
Explain any missing or insufficient evidence.

Metadata alone is not evidence of performance. A disclosed risk does not automatically mean poor performance. Do not claim to have reviewed sections not present in this excerpt.

Filing excerpt:
{content}
"""
    return run_groq(prompt)

def extract_label(text, section, allowed):
    match = re.search(
        rf"{re.escape(section)}\s*:\s*(.*?)(?=\n\s*[A-Z][A-Z ]+:\s*|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return "Insufficient Data"

    section_text = match.group(1).strip()
    first_line = section_text.splitlines()[0].strip() if section_text else ""

    for label in allowed:
        if re.match(rf"^{re.escape(label)}\b", first_line, re.IGNORECASE):
            return label

    for label in allowed:
        if re.search(rf"\b{re.escape(label)}\b", first_line, re.IGNORECASE):
            return label

    return "Insufficient Data"

def compare_years(results):
    summaries = "\n\n".join(
        f"Fiscal year: {item['year']}\n"
        f"Investment signal: {item['sentiment']}\n"
        f"Management tone: {item['management_tone']}\n"
        f"Analysis: {item['ai_analysis'][:700]}"
        for item in results
    )
    return run_groq(f"""
Compare these SEC 10-K analyses chronologically.
Explain meaningful changes in reported performance, risks, and management tone only when supported by evidence. Do not invent trends. Mention missing evidence and keep the comparison concise.

{summaries}
""")

def answer_question(question, all_content):
    return run_groq(f"""
Answer using only the supplied SEC filing excerpts.
Question: {question}
Do not invent facts or figures. Mention the fiscal year when possible. If the answer is absent, say so. Keep the answer concise.

Filing excerpts:
{all_content[:6000]}
""")

st.markdown("""
<style>
.header{text-align:center;padding:1.5rem 0 1rem}
.header h1{font-size:2.8rem;font-weight:800;color:#1a3f6f;margin-bottom:.3rem}
.header h1 span{color:#c9a84c}
.header p{font-size:1rem;color:#4a5568}
</style>
<div class="header">
<h1>SEC<span>Insight</span></h1>
<p>AI-powered analysis of SEC 10-K filings, risks and year-over-year trends</p>
</div>
""", unsafe_allow_html=True)

st.divider()

defaults = {
    "results": None,
    "all_content": None,
    "ticker_analyzed": None,
    "comparison": None,
    "answer": None,
}
for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

ticker = st.text_input(
    "Stock Ticker",
    placeholder="Enter ticker e.g. AAPL, TSLA, MSFT",
).strip().upper()

years = st.slider("Years to analyze", 1, 5, 3, key="years_to_analyze")
analyze_button = st.button("Analyze", use_container_width=True)

if analyze_button:
    for key in defaults:
        st.session_state[key] = None

    if not ticker:
        st.error("Please enter a ticker symbol.")
    elif client is None:
        st.error("Groq API key not found. Add GROQ_API_KEY to Streamlit Cloud Secrets.")
    else:
        try:
            with st.status("Retrieving SEC filings...", expanded=True) as progress:
                progress.write(f"Ticker: {ticker}")
                progress.write(f"Requested fiscal years: {years}")
                filings = download_filings(ticker, years)
                progress.write(f"Retrieved {len(filings)} usable filings.")
                progress.update(label="SEC filing retrieval complete", state="complete")

            if not filings:
                st.error("No usable 10-K filings were found.")
            else:
                filings = sorted(filings, key=lambda item: int(item["year"]))[-years:]
                if len(filings) < years:
                    st.warning(f"Requested {years} years, but only {len(filings)} distinct fiscal years were found.")

                results = []
                content_parts = []
                failed = False

                for filing in filings:
                    filing["ticker"] = ticker
                    with st.spinner(f"Analyzing fiscal year {filing['year']}..."):
                        analysis = analyze_with_ai(filing)

                    if not analysis:
                        failed = True
                        break

                    sentiment = extract_label(
                        analysis,
                        "INVESTMENT SIGNAL",
                        ["Positive", "Neutral", "Negative", "Insufficient Data"],
                    )
                    tone = extract_label(
                        analysis,
                        "MANAGEMENT TONE",
                        ["Optimistic", "Neutral", "Cautious", "Insufficient Data"],
                    )

                    results.append({
                        "year": int(filing["year"]),
                        "sentiment": sentiment,
                        "management_tone": tone,
                        "ai_analysis": analysis,
                        "filed_date": filing.get("filed_date", ""),
                    })

                    excerpt = filing.get("analysis_content") or filing.get("content", "")
                    content_parts.append(
                        f"\nFiscal year {filing['year']}:\n{excerpt[:1500]}"
                    )

                if failed:
                    st.error("An AI analysis failed. Review the error above and try again.")
                elif results:
                    with st.spinner("Comparing fiscal years..."):
                        comparison = compare_years(results)

                    if comparison:
                        st.session_state.results = results
                        st.session_state.all_content = "\n".join(content_parts)
                        st.session_state.ticker_analyzed = ticker
                        st.session_state.comparison = comparison
                    else:
                        st.error("The yearly analysis completed, but the comparison failed.")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            with st.expander("Technical details"):
                st.exception(exc)

if st.session_state.results:
    results = sorted(st.session_state.results, key=lambda item: item["year"])
    all_content = st.session_state.all_content
    ticker_analyzed = st.session_state.ticker_analyzed

    st.success(f"Analysis complete for {ticker_analyzed}")
    st.divider()

    latest = results[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Company", ticker_analyzed)
    col2.metric("Years Analyzed", len(results))
    col3.metric("Latest Sentiment", latest["sentiment"])

    if all(item["sentiment"] == "Insufficient Data" for item in results):
        st.warning("No reliable investment signals were extracted. Check the AI analysis and the filing text. Do not interpret this as a negative or positive signal.")

    st.divider()
    st.subheader("Yearly Analysis")
    for result in results:
        with st.expander(f"{result['year']} — {result['sentiment']}", expanded=True):
            st.caption(f"Management tone: {result['management_tone']}")
            st.markdown(result["ai_analysis"])

    st.divider()
    st.subheader("Year-over-Year Comparison")
    st.write(st.session_state.comparison)

    st.divider()
    st.subheader("Ask a Question About These Filings")
    with st.form("question_form"):
        question = st.text_input("Your question", placeholder="e.g. What are the main revenue streams?")
        ask_button = st.form_submit_button("Ask")

    if ask_button:
        if not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Analyzing the supplied excerpts..."):
                answer = answer_question(question.strip(), all_content)
            if answer:
                st.session_state.answer = answer

    if st.session_state.answer:
        st.info(st.session_state.answer)

    st.divider()
    st.subheader("Results Table")
    df = pd.DataFrame([
        {
            "Fiscal Year": item["year"],
            "Management Tone": item["management_tone"],
            "Investment Signal": item["sentiment"],
            "Filing Date": item["filed_date"],
        }
        for item in results
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Results as CSV",
        data=df.to_csv(index=False),
        file_name=f"{ticker_analyzed}_analysis.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Sentiment Trend")
    sentiment_scores = {
        "Negative": -1,
        "Neutral": 0,
        "Positive": 1,
        "Insufficient Data": float("nan"),
    }
    plot_df = pd.DataFrame([
        {"year": item["year"], "score": sentiment_scores[item["sentiment"]]}
        for item in results
    ]).dropna(subset=["score"])

    if plot_df.empty:
        st.warning("No reliable sentiment scores are available, so the graph cannot be plotted yet.")
    else:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(plot_df["year"], plot_df["score"], marker="o", linewidth=2)
        ax.set_yticks([-1, 0, 1], ["Negative", "Neutral", "Positive"])
        ax.set_ylim(-1.3, 1.3)
        ax.set_xlabel("Fiscal Year")
        ax.set_title(f"{ticker_analyzed} — Filing Sentiment")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    st.caption("Sentiment is an AI-generated interpretation of the supplied filing excerpt, not a prediction of stock returns.")
