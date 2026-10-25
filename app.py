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
        )
        return response.choices[0].message.content
    except GroqError as exc:
        status = getattr(exc, "status_code", None)
        if status == 401:
            message = "Invalid Groq API key."
        elif status == 404:
            message = f"Model '{MODEL_ID}' was not found."
        elif status == 413:
            message = "The request is too large. Reduce the filing text sent to Groq."
        elif status == 429:
            message = "Groq usage or rate limit reached."
        else:
            message = f"Groq request failed (HTTP {status or 'unknown'})."
        st.error(message)
        return None
    except Exception as exc:
        st.error(f"Unexpected AI error: {exc}")
        return None

def analyze_with_ai(filing):
    content = filing.get("analysis_content") or filing.get("content", "")
    content = content[:4000]
    prompt = f"""
You are a careful financial analyst reviewing a company's SEC 10-K.
Ticker: {filing.get('ticker', 'Unknown')}
Fiscal year: {filing['year']}

Analyze only the supplied filing text. Be concise and do not invent facts.

Return exactly these sections:

PERFORMANCE SUMMARY:
Write two evidence-based sentences.

TOP THREE RISKS:
List up to three risks supported by the text. If evidence is insufficient, say so.

MANAGEMENT TONE:
Choose exactly one: Optimistic, Neutral, Cautious, or Insufficient Data. Explain briefly.

INVESTMENT SIGNAL:
Choose exactly one: Positive, Neutral, Negative, or Insufficient Data. This is a filing-based assessment, not personalized investment advice. Do not infer a positive signal from positive words alone.

EVIDENCE LIMITATIONS:
State what missing information limits the analysis.

Rules:
- Filing metadata alone is not evidence of company performance.
- Do not invent financial figures, risks, or management statements.
- Distinguish reported results from forward-looking statements.
- If substantive evidence is insufficient, explicitly say so.
- Begin each section with its exact heading.
- Put the selected label immediately after MANAGEMENT TONE: and INVESTMENT SIGNAL:.

Filing text:
{content}
"""
    return run_groq(prompt)

def extract_label(text, section, allowed):
    match = re.search(
        rf"{re.escape(section)}\s*:\s*(.*?)(?=\n[A-Z][A-Z ]+:\s*|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return "Insufficient Data"
    section_text = match.group(1).strip()
    first_line = section_text.splitlines()[0] if section_text else ""
    for label in allowed:
        if re.search(rf"^\s*{re.escape(label)}\b", first_line, re.IGNORECASE):
            return label
    return "Insufficient Data"

def compare_years(results):
    summaries = "\n\n".join(
        f"Fiscal year: {item['year']}\n"
        f"Investment signal: {item['sentiment']}\n"
        f"Management tone: {item['management_tone']}\n"
        f"Analysis:\n{item['ai_analysis'][:1000]}"
        for item in results
    )
    prompt = f"""
Compare these SEC 10-K analyses chronologically.
Explain meaningful changes in reported performance, disclosed risks, and management tone only where evidence supports them. Do not invent trends or treat missing data as stability. Mention limitations when evidence is insufficient. Keep the response concise.

Analyses:
{summaries}
"""
    return run_groq(prompt)

def answer_question(question, all_content):
    excerpts = all_content[:4000]
    prompt = f"""
Answer the question using only the supplied SEC filing excerpts.

Question: {question}

Rules:
- Cite the fiscal year when possible.
- Do not invent facts or figures.
- If the excerpts do not contain the answer, say so.
- Distinguish facts from interpretation.
- Keep the answer concise.

Filing excerpts:
{excerpts}
"""
    return run_groq(prompt)

st.markdown("""
<style>
.header{text-align:center;padding:1.5rem 0 1rem 0}
.header h1{font-size:2.8rem;font-weight:800;color:#1a3f6f;margin-bottom:0.3rem}
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

ticker = st.text_input("Stock Ticker", placeholder="Enter ticker e.g. AAPL, TSLA, MSFT").strip().upper()
years = st.slider("Years to analyze", min_value=1, max_value=5, value=3, key="years_to_analyze")
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
            with st.status("Starting SEC filing retrieval...", expanded=True) as progress:
                progress.write(f"Ticker: {ticker}")
                progress.write(f"Requested fiscal years: {years}")
                progress.update(label="Contacting SEC and retrieving filings...")
                filings = download_filings(ticker, years)
                progress.write(f"Retrieved {len(filings)} usable fiscal years.")
                progress.update(label="SEC filing retrieval complete", state="complete")

            if not filings:
                st.error("No usable 10-K filings were found. Check the ticker and SEC access.")
            else:
                filings = sorted(filings, key=lambda item: int(item["year"]))[-years:]
                if len(filings) < years:
                    st.warning(f"Requested {years} years, but only {len(filings)} distinct usable fiscal years were found.")

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
                    management_tone = extract_label(
                        analysis,
                        "MANAGEMENT TONE",
                        ["Optimistic", "Neutral", "Cautious", "Insufficient Data"],
                    )
                    results.append({
                        "year": filing["year"],
                        "sentiment": sentiment,
                        "management_tone": management_tone,
                        "ai_analysis": analysis,
                        "filed_date": filing.get("filed_date", ""),
                    })
                    content_parts.append(
                        f"\n\n--- Fiscal year {filing['year']} ---\n"
                        f"{(filing.get('analysis_content') or filing.get('content', ''))[:4000]}"
                    )

                if failed:
                    st.error("An AI analysis failed. Check the error above and try again.")
                elif results:
                    with st.spinner("Comparing fiscal years..."):
                        comparison = compare_years(results)
                    if comparison:
                        st.session_state.results = results
                        st.session_state.all_content = "\n".join(content_parts)
                        st.session_state.ticker_analyzed = ticker
                        st.session_state.comparison = comparison
                    else:
                        st.error("The yearly analysis completed, but the comparison failed. Please try again.")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            with st.expander("Technical details"):
                st.exception(exc)

if st.session_state.results:
    results = st.session_state.results
    all_content = st.session_state.all_content
    ticker_analyzed = st.session_state.ticker_analyzed
    comparison = st.session_state.comparison

    st.success(f"Analysis complete for {ticker_analyzed}")
    st.divider()
    latest = max(results, key=lambda item: int(item["year"]))
    col1, col2, col3 = st.columns(3)
    col1.metric("Company", ticker_analyzed)
    col2.metric("Years Analyzed", len(results))
    col3.metric("Latest Sentiment", latest["sentiment"])

    st.divider()
    st.subheader("Yearly Analysis")
    for result in sorted(results, key=lambda item: int(item["year"])):
        with st.expander(f"{result['year']} — {result['sentiment']}", expanded=True):
            st.caption(f"Management tone: {result['management_tone']}")
            st.markdown(result["ai_analysis"])

    st.divider()
    st.subheader("Year-over-Year Comparison")
    st.write(comparison)

    st.divider()
    st.subheader("Ask a Question About These Filings")
    st.caption(f"Ask questions about {ticker_analyzed}'s SEC filings.")
    with st.form("question_form"):
        question = st.text_input("Your question", placeholder="e.g. What are the main revenue streams?")
        ask_button = st.form_submit_button("Ask")
    if ask_button:
        if not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Analyzing the supplied filing excerpts..."):
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
        for item in sorted(results, key=lambda item: int(item["year"]))
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download Results as CSV",
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
        {"year": int(item["year"]), "score": sentiment_scores[item["sentiment"]]}
        for item in sorted(results, key=lambda item: int(item["year"]))
    ])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(plot_df["year"], plot_df["score"], marker="o", linewidth=2)
    ax.set_yticks([-1, 0, 1], ["Negative", "Neutral", "Positive"])
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("Fiscal Year")
    ax.set_title(f"{ticker_analyzed} — Filing Sentiment")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Sentiment is an AI-generated interpretation of the supplied filing text, not a validated prediction of stock returns.")
