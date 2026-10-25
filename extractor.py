import os
import re
from pathlib import Path
from datetime import date

import requests
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

BASE_DIR = Path("sec-edgar-filings")
SEC_EMAIL = os.getenv("SEC_USER_AGENT_EMAIL", "9f06eshaseloth@gmail.com")
SEC_HEADERS = {"User-Agent": f"SECInsight contact {SEC_EMAIL}", "Accept-Encoding": "gzip, deflate"}


def clean_text(raw):
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def get_document(raw):
    """Prefer the actual 10-K document inside an SEC complete-submission text file."""
    documents = re.findall(r"(?is)<DOCUMENT>(.*?)</DOCUMENT>", raw)
    for document in documents:
        match = re.search(r"(?im)<TYPE>\s*([^\s<]+)", document)
        if match and match.group(1).upper() == "10-K":
            body = re.search(r"(?is)<TEXT>(.*?)</TEXT>", document)
            return body.group(1) if body else document
    return raw


def _section(text, start_pattern, end_pattern, limit=9000):
    starts = list(re.finditer(start_pattern, text, re.I))
    ends = list(re.finditer(end_pattern, text, re.I))
    candidates = []
    for start in starts:
        end = next((e for e in ends if e.start() > start.end()), None)
        if not end:
            continue
        candidate = text[start.start():end.start()].strip()
        # Real filing sections are normally much longer than TOC entries.
        if len(candidate) >= 500:
            candidates.append(candidate)
    if not candidates:
        return ""
    return max(candidates, key=len)[:limit]


def extract_sections(raw_html):
    text = clean_text(raw_html)
    item = r"\bITEM\s*"
    business = _section(
        text,
        item + r"1\s*[.\-:]?\s+BUSINESS\b",
        item + r"1A\s*[.\-:]?\s+RISK\s+FACTORS\b",
        5000,
    )
    risks = _section(
        text,
        item + r"1A\s*[.\-:]?\s+RISK\s+FACTORS\b",
        item + r"(?:1B\b|2\s*[.\-:]?\s+PROPERTIES\b)",
        6500,
    )
    management = _section(
        text,
        item + r"7\s*[.\-:]?\s+MANAGEMENT'?S?\s+DISCUSSION\s+AND\s+ANALYSIS\b",
        item + r"(?:7A\b|8\s*[.\-:]?\s+FINANCIAL\s+STATEMENTS\b)",
        9000,
    )
    parts = []
    for title, body in (
        ("BUSINESS", business),
        ("RISK FACTORS", risks),
        ("MANAGEMENT DISCUSSION AND ANALYSIS", management),
    ):
        if len(body) >= 500:
            parts.append(f"{title}\n{body}")
    return "\n\n".join(parts), text


def _sec_get(url):
    response = requests.get(url, headers=SEC_HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def get_companyfacts(ticker, years):
    """Retrieve reported annual financial facts directly from the SEC CompanyFacts API."""
    ticker = ticker.strip().upper()
    ticker_data = _sec_get("https://www.sec.gov/files/company_tickers.json")
    company = next(
        (item for item in ticker_data.values()
         if str(item.get("ticker", "")).upper() == ticker),
        None,
    )
    if not company:
        raise ValueError(f"Ticker {ticker} was not found in the SEC ticker list.")

    cik = int(company["cik_str"])
    facts_json = _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
    usgaap = facts_json.get("facts", {}).get("us-gaap", {})

    metric_tags = {
        "Revenue": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "Revenues",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ],
        "Net income": ["NetIncomeLoss", "ProfitLoss"],
        "Operating cash flow": ["NetCashProvidedByUsedInOperatingActivities"],
    }

    def annual_value(tags, year):
        for tag in tags:
            fact = usgaap.get(tag)
            if not fact:
                continue
            for unit, entries in fact.get("units", {}).items():
                valid = []
                for entry in entries:
                    if entry.get("form") != "10-K":
                        continue
                    if not entry.get("end", "").startswith(str(year)):
                        continue
                    start = entry.get("start")
                    if not start:
                        continue
                    try:
                        duration = (date.fromisoformat(entry["end"]) - date.fromisoformat(start)).days
                    except (ValueError, KeyError):
                        continue
                    if not 330 <= duration <= 400:
                        continue
                    valid.append(entry)
                if valid:
                    # Prefer the most recently filed annual value for this fiscal year.
                    chosen = max(valid, key=lambda x: x.get("filed", ""))
                    return float(chosen["val"]), unit, tag
        return None, None, None

    output = {}
    for year in years:
        row = {"year": int(year)}
        for label, tags in metric_tags.items():
            value, unit, tag = annual_value(tags, int(year))
            row[label] = value
            row[f"{label} unit"] = unit
            row[f"{label} tag"] = tag
        output[int(year)] = row
    return output


def download_filings(ticker, years):
    ticker = ticker.strip().upper()
    years = int(years)
    if not re.fullmatch(r"[A-Z0-9.-]{1,10}", ticker):
        raise ValueError("Enter a valid stock ticker.")
    if not 1 <= years <= 5:
        raise ValueError("Select between 1 and 5 years.")

    downloader = Downloader("SECInsight", SEC_EMAIL)
    downloader.get("10-K", ticker, limit=years)

    filing_dir = BASE_DIR / ticker / "10-K"
    if not filing_dir.exists():
        raise FileNotFoundError(f"SEC filing directory not found: {filing_dir.resolve()}")

    candidates = []
    for folder in filing_dir.iterdir():
        if not folder.is_dir():
            continue
        txt_files = list(folder.glob("*.txt"))
        raw = ""
        file_path = None
        if txt_files:
            file_path = txt_files[0]
            try:
                raw = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

        period = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", raw, re.I)
        filed = re.search(r"FILED AS OF DATE:\s*(\d{8})", raw, re.I)
        if not period:
            # Some download versions don't preserve the metadata in the same text file.
            continue

        # Prefer the primary HTML filing over the complete-submission .txt.
        html_files = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".htm", ".html"}
            and not p.name.lower().startswith(("ex", "xsl", "r"))
        ]
        if html_files:
            primary = max(html_files, key=lambda p: p.stat().st_size)
            try:
                source_html = primary.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                source_html = get_document(raw)
        else:
            source_html = get_document(raw)

        sections, full_text = extract_sections(source_html)
        report_year = int(period.group(1)[:4])
        candidates.append({
            "year": report_year,
            "filed_date": filed.group(1) if filed else "",
            "content": full_text,
            "analysis_content": sections,
            "source_file": str(file_path) if file_path else str(folder),
        })

    candidates.sort(key=lambda item: item["filed_date"], reverse=True)
    selected, seen = [], set()
    for filing in candidates:
        if filing["year"] in seen:
            continue
        seen.add(filing["year"])
        selected.append(filing)
        if len(selected) >= years:
            break
    selected.sort(key=lambda item: item["year"])
    if not selected:
        raise RuntimeError("No 10-K filings with reporting-period metadata were found.")

    # Financial facts are an independent, structured source, not an AI estimate.
    try:
        facts = get_companyfacts(ticker, [f["year"] for f in selected])
    except Exception as exc:
        facts = {}
        facts_error = str(exc)
    else:
        facts_error = ""

    for filing in selected:
        filing["financials"] = facts.get(filing["year"], {"year": filing["year"]})
        filing["financials_error"] = facts_error
    return selected
