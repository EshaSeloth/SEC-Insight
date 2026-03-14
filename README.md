# SECInsight

SECInsight is an AI-powered financial analysis tool that automates the process of reading and interpreting SEC 10-K filings. Public company filings run into hundreds of pages and are traditionally analyzed manually by financial analysts and investors. SECInsight reduces that process to seconds.

The tool pulls real filings directly from SEC EDGAR, performs sentiment analysis, extracts key risks, evaluates management tone and tracks how a company's financial narrative has evolved year over year. Users can also ask natural language questions about any filing and receive answers grounded in the actual document content.

Built for traders, investors, analysts and anyone in the financial sector who wants to make faster and more informed decisions about a company.

## Features

- Automated retrieval of 10-K filings from SEC EDGAR for any US public company
- Year over year sentiment analysis using financial signal scoring
- AI generated summaries covering company performance, top risks and management tone per year
- Temporal comparison highlighting how a company has changed across filings
- Natural language question answering grounded in actual filing content
- Sentiment trend visualization across years
- CSV export of all results

## Tech Stack

- Python
- Streamlit
- Groq LLM
- SEC EDGAR Downloader
- Pandas
- Matplotlib
- Docker

## Testing

Tested across 5 major US public companies over 3 years of filings each:
- Apple (AAPL)
- Tesla (TSLA)
- Microsoft (MSFT)
- Google (GOOGL)
- Amazon (AMZN)

## Results

Sentiment scoring and AI analysis correctly identified Apple and Microsoft as consistently positive across all years analyzed. Tesla showed more volatile sentiment reflecting its mixed financial disclosures. Risk language increased across all five companies in more recent filings. Management tone shifted from optimistic to neutral post 2023 for the majority of companies tested.    ss