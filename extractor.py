progress = st.status("Starting SEC filing retrieval...", expanded=True)

try:
    progress.write(f"Ticker: {ticker}")
    progress.write(f"Requested fiscal years: {years}")

    progress.update(label="Contacting SEC and retrieving filings...")

    filings = download_filings(ticker, years)

    progress.write(f"Retrieved {len(filings)} usable fiscal years.")
    progress.update(
        label="SEC filing retrieval complete",
        state="complete",
    )

except Exception as exc:
    progress.update(
        label="SEC filing retrieval failed",
        state="error",
    )
    st.exception(exc)
    filings = None
