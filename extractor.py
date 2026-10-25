import re
from pathlib import Path

from sec_edgar_downloader import Downloader

BASE_DIR = Path("sec-edgar-filings")


def clean_text(raw):
    raw = re.sub(r"(?is)<script\b.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style\b.*?</style>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def download_filings(ticker, years):
    ticker = ticker.strip().upper()
    years = int(years)

    if not ticker or not 1 <= years <= 5:
        raise ValueError("Enter a valid ticker and select 1-5 years.")

    downloader = Downloader(
        "SECInsight",
        "your-real-email@example.com",
    )

    downloader.get("10-K", ticker, limit=years)

    filing_dir = BASE_DIR / ticker / "10-K"

    if not filing_dir.exists():
        raise FileNotFoundError(
            f"SEC download directory not found: {filing_dir.resolve()}"
        )

    candidates = []

    for folder in filing_dir.iterdir():
        if not folder.is_dir():
            continue

        txt_files = list(folder.glob("*.txt"))
        if not txt_files:
            continue

        file_path = txt_files[0]

        try:
            raw = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except OSError:
            continue

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

        if not period:
            continue

        year = int(period.group(1)[:4])
        filed_date = filed.group(1) if filed else "00000000"
        content = clean_text(raw)

        if len(content) < 1000:
            continue

        candidates.append({
            "year": year,
            "filed_date": filed_date,
            "content": content[:30000],
            "analysis_content": content[:22000],
        })

    candidates.sort(
        key=lambda item: item["filed_date"],
        reverse=True,
    )

    selected = []
    seen_years = set()

    for filing in candidates:
        if filing["year"] in seen_years:
            continue

        seen_years.add(filing["year"])
        selected.append(filing)

        if len(selected) >= years:
            break

    selected.sort(key=lambda item: item["year"])

    if not selected:
        raise RuntimeError(
            "No usable 10-K filings were extracted. "
            "Check the SEC download and filing text."
        )

    return selected
