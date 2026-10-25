from pathlib import Path

code = r'''import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq, GroqError

from extractor import download_filings

load_dotenv()

# Streamlit Cloud: store GROQ_API_KEY in App Settings > Secrets.
# Local development: optionally store it in a .env file.
try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    api_key = None

api_key = api_key or os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

# Use one model constant so all AI features use the same model.
MODEL_ID = "openai/gpt-oss-120b"


def run_groq(prompt):
    """Run a Groq completion and show a readable error instead of crashing."""
    if client is None:
        st.error(
            "Groq API key not found. Add GROQ_API_KEY to Streamlit Cloud "
            "Secrets, or to a local .env file."
        )
        return None

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except GroqError as exc:
        status_code = getattr(exc, "status_code", None)
        if status_code == 401:
            message = "Groq rejected the API key. Check GROQ_API_KEY in Streamlit Secrets."
        elif status_code == 404:
            message = (
                f"Groq returned Not Found for model '{MODEL_ID}'. "
                "Check the current model ID in Groq Console."
            )
        elif status_code == 429:
            message = "Groq rate limit or usage limit reached. Check your Groq Console limits."
        else:
            message = f"Groq API request failed (HTTP {status_code or 'unknown'}). Check Streamlit Cloud Logs."
        st.error(message)
        return None
    except Exception:
        st.error("The AI request failed unexpectedly. Check Streamlit Cloud Logs for details.")
        return None


def analyze_with_ai(content):
    prompt = f"""You are a financial analyst. Analyze this SEC 10-K filing and provide:
1. A 2 sentence summary of company performance
2. Top 3 risks mentioned
3. Management tone: Optimistic, Neutral or Cautious
4. Overall investment signal: Positive, Neutral or Negative

Filing content:
{content[:3000]}

Keep your response concise and professional."""
    return run_groq(prompt)


def compare_years(results):
    prompt = f"""You are a financial analyst. Based on these yearly SEC filing analyses, describe how this company has changed over the years. Has the tone improved or worsened? Are risks increasing or decreasing? Keep it to 3 sentences.

Yearly analyses:
{results}"""
    return run_groq(prompt)


def answer_question(question, all_content):
    prompt = f"""You are a financial analyst. Answer the following question based only on the SEC 10-K filing content provided. Be concise and specific.

Question: {question}

Filing content:
{all_content[:6000]}"""
    return run_groq(prompt)


st.set_page_config(page_title="SECInsight", layout="centered")

st.markdown(
    """
    <style>
    .header { text-align: center; padding: 2rem 0 1rem 0; }
    .header h1 { font-size: 2.8rem; font-weight: 800; color: #1a3f6f; margin-bottom: 0.3rem; }
    .header h1 span { color: #c9a84c; }
    .header p { font-size: 1rem; color: #4a5568; margin: 0; }
    </style>
    <div class='header'>
        <h1>SEC<span>Insight</span></h1>
        <p>AI-powered analysis of SEC 10-K filings, sentiment, risks and year over year trends</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# Session state setup
if "results" not in st.session_state:
    st.session_state.results = None
if "all_content" not in st.session_state:
    st.session_state.all_content = None
if "yearly_summaries" not in st.session_state:
    st.session_state.yearly_summaries = None
if "ticker_analyzed" not in st.session_state:
    st.session_state.ticker_analyzed = None
if "comparison" not in st.session_state:
    st.session_state.comparison = None
if "answer" not in st.session_state:
    st.session_state.answer = None

ticker = st.text_input("Stock Ticker", placeholder="Enter ticker e.g. AAPL, TSLA, MSFT").upper()
years = st.slider("Years to analyze", 1, 5, 3)
analyze_button = st.button("Analyze", use_container_width=True)

if analyze_button:
    if not ticker:
        st.error("Please enter a ticker symbol.")
    else:
        try:
            with st.spinner("Downloading SEC filings..."):
                filings = download_filings(ticker, years)
        except Exception:
            st.error("Could not download SEC filings. Check the ticker and extractor logs.")
            filings = None

        if not filings:
            st.error("No filings found for this ticker, or the SEC filing download failed.")
        else:
            positive = [
                "increase", "increased", "growth", "grew", "strong",
                "record", "higher", "improved", "expansion", "gained",
                "profitable", "success", "opportunity", "favorable", "up"
            ]
            negative = [
                "risk", "loss", "decline", "decreased", "uncertainty",
                "lawsuit", "adverse", "unfavorable", "impairment", "down",
                "weak", "difficult", "challenge", "volatile", "deficit"
            ]

            results = []
            yearly_summaries = ""
            all_content = ""
            ai_failed = False

            for filing in filings:
                content = filing["content"]
                with st.spinner(f"Analyzing {filing['year']}..."):
                    ai_analysis = analyze_with_ai(content)

                if ai_analysis is None:
                    ai_failed = True
                    break

                pos_score = sum(content.lower().count(w) for w in positive)
                neg_score = sum(content.lower().count(w) for w in negative)

                if pos_score > neg_score:
                    sentiment = "Positive"
                elif neg_score > pos_score:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"

                results.append({
                    "year": filing["year"],
                    "positive_signals": pos_score,
                    "negative_signals": neg_score,
                    "sentiment": sentiment,
                    "ai_analysis": ai_analysis,
                })

                yearly_summaries += f"\n{filing['year']}: {ai_analysis}\n"
                all_content += f"\n\n--- {filing['year']} ---\n{content}"

            if not ai_failed:
                with st.spinner("Generating year-over-year comparison..."):
                    comparison = compare_years(yearly_summaries)

                if comparison is not None:
                    st.session_state.results = results
                    st.session_state.all_content = all_content
                    st.session_state.yearly_summaries = yearly_summaries
                    st.session_state.ticker_analyzed = ticker
                    st.session_state.comparison = comparison
                    st.session_state.answer = None

if st.session_state.results:
    results = st.session_state.results
    all_content = st.session_state.all_content
    ticker_analyzed = st.session_state.ticker_analyzed
    comparison = st.session_state.comparison

    st.success(f"Analysis complete for {ticker_analyzed}")
    st.divider()

    latest = results[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Company", ticker_analyzed)
    col2.metric("Years Analyzed", len(results))
    col3.metric("Latest Sentiment", latest["sentiment"])

    st.divider()
    st.subheader("Yearly Analysis")

    for result in results:
        with st.expander(f"{result['year']} — {result['sentiment']}", expanded=True):
            col1, col2 = st.columns(2)
            col1.metric("Positive Signals", result["positive_signals"])
            col2.metric("Negative Signals", result["negative_signals"])
            st.info(result["ai_analysis"])

    st.divider()
    st.subheader("Year-over-Year Comparison")
    st.success(comparison)

    st.divider()
    st.subheader("Ask a Question About These Filings")
    st.caption(f"Ask anything about {ticker_analyzed} SEC filings and AI will answer from the provided documents.")

    question = st.text_input("Your question", placeholder="e.g. What are the main revenue streams?")

    if st.button("Ask", use_container_width=True):
        if not question:
            st.error("Please enter a question.")
        else:
            with st.spinner("Finding answer from filings..."):
                answer = answer_question(question, all_content)
            if answer is not None:
                st.session_state.answer = answer

    if st.session_state.answer:
        st.info(st.session_state.answer)

    st.divider()
    st.subheader("Results Table")
    df = pd.DataFrame(results)[["year", "positive_signals", "negative_signals", "sentiment"]]
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"{ticker_analyzed}_analysis.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Sentiment Trend")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["year"], df["positive_signals"], label="Positive", color="#1a3f6f", marker="o", linewidth=2)
    ax.plot(df["year"], df["negative_signals"], label="Negative", color="#c0392b", marker="o", linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Signal Count")
    ax.set_title(f"{ticker_analyzed} - Sentiment Over Time", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
'''
path = Path("/mnt/data/app.py")
path.write_text(code, encoding="utf-8")
print(f"Created {path} ({len(code.splitlines())} lines)")

