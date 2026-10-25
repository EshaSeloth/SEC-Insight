
import os
import re
from pathlib import Path

from sec_edgar_downloader import Downloader


BASE_DIR = Path("sec-edgar-filings")


def clean_text(raw):
    """Remove HTML tags and normalize whitespace."""
    text = re.sub(r"(?is)<script\b.*?</script>", " ", raw)
    text = re.sub(r"(?is)<style\b.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_metadata(raw, folder_name):
    """Extract fiscal period and filing date from SEC headers."""
    period = re.search(
        r"CONFORMED PERIOD OF REPORT:\s*(\d{8})",
        raw,
        re.IGNORECASE,
    )

    filed = re.search(
        r"FILED AS OF DATE:\s*(\d{8})",
        raw,
        re.IGNORECASE,
    )

    report_date = period.group(1) if period else None
    filed_date = filed.group(1) if filed else None
    if not report_date:
        accession_year = re.search(r"-(\d{2})-", folder_name)
        if accession_year:
            report_year = int(accession_year.group(1))
            report_year += 2000 if report_year < 80 else 1900
        else:
            report_year = None
    else:
        report_year = int(report_date[:4])

    return report_year, filed_date


def make_analysis_excerpt(cleaned):
    """
    Prioritize Risk Factors, MD&A, and financial statements.
    Search from the end to reduce the chance of selecting
    the table-of-contents entry instead of the real section.
    """
    sections = [
        (r"\bitem\s+1a\b", "Risk Factors"),
        (r"\bitem\s+7\b", "Management Discussion and Analysis"),
        (r"\bitem\s+8\b", "Financial Statements"),
    ]

    excerpts = []

    for pattern, label in sections:
        matches = list(
            re.finditer(pattern, cleaned, re.IGNORECASE)
        )

        if not matches:
            continue
        match = matches[-1]
        start = match.start()
        excerpt = cleaned[start:start + 10000]

        excerpts.append(
            f"\n\n--- {label} ---\n{excerpt}"
        )

    if excerpts:
        return "\n".join(excerpts)[:25000]
    return cleaned[:25000]


def download_filings(ticker, years):
    ticker = ticker.strip().upper()
    years = int(years)

    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    if years < 1 or years > 5:
        raise ValueError("Years must be between 1 and 5.")

    downloader = Downloader(
        "SECInsight",
        "your-real-email@example.com",
    )
    downloader.get("10-K", ticker, limit=years + 5)

    filing_dir = BASE_DIR / ticker / "10-K"

    if not filing_dir.exists():
        return []

    candidates = []

    for folder in filing_dir.iterdir():
        if not folder.is_dir():
            continue

        for file_path in folder.glob("*.txt"):
            try:
                raw = file_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except OSError:
                continue

            if not raw.strip():
                continue

            report_year, filed_date = extract_metadata(
                raw,
                folder.name,
            )

            if not report_year:
                continue

            cleaned = clean_text(raw)

            if len(cleaned) < 1000:
                continue

            candidates.append({
                "year": report_year,
                "filed_date": filed_date or "00000000",
                "content": cleaned[:50000],
                "analysis_content": make_analysis_excerpt(cleaned),
                "accession": folder.name,
            })
            break
    candidates.sort(
        key=lambda item: (
            item["filed_date"],
            item["year"],
        ),
        reverse=True,
    )
    unique_years = []
    seen_years = set()
    for filing in candidates:
        if filing["year"] in seen_years:
            continue
        seen_years.add(filing["year"])
        unique_years.append(filing)

        if len(unique_years) == years:
            break
    unique_years.sort(key=lambda item: item["year"])
    return unique_years
