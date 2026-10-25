
import re
from pathlib import Path
from sec_edgar_downloader import Downloader

BASE_DIR = Path("sec-edgar-filings")

def clean_text(raw):
    raw = re.sub(r"(?is)<script\b.*?</script>", " ", raw)
    raw = re.sub(r"(?is)<style\b.*?</style>", " ", raw)
    raw = re.sub(r"(?is)<!--.*?-->", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = re.sub(r"&nbsp;|&#160;", " ", raw, flags=re.I)
    raw = re.sub(r"&amp;", "&", raw, flags=re.I)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()

def get_document(raw):
    documents = re.findall(r"(?is)<DOCUMENT>(.*?)</DOCUMENT>", raw)
    for document in documents:
        match = re.search(r"(?im)<TYPE>\s*([^\s<]+)", document)
        if match and match.group(1).upper() == "10-K":
            body = re.search(r"(?is)<TEXT>(.*?)</TEXT>", document)
            return body.group(1) if body else document
    return raw

def find_best_section(text, start_pattern, end_pattern, limit=3500):
    starts = list(re.finditer(start_pattern, text, re.I))
    ends = list(re.finditer(end_pattern, text, re.I))
    best = ""
    for start in starts:
        end = next((item for item in ends if item.start() > start.end()), None)
        if end:
            candidate = text[start.start():end.start()].strip()
            if len(candidate) > len(best):
                best = candidate
    return best[:limit]

def extract_sections(text):
    item = r"\bITEM\s+"
    business = find_best_section(
        text,
        item + r"1\s*[.\-:]?\s+BUSINESS\b",
        item + r"1A\s*[.\-:]?\s+RISK\s+FACTORS\b"
    )
    risks = find_best_section(
        text,
        item + r"1A\s*[.\-:]?\s+RISK\s+FACTORS\b",
        item + r"(?:1B\b|2\s*[.\-:]?\s+PROPERTIES\b)"
    )
    management = find_best_section(
        text,
        item + r"7\s*[.\-:]?\s+MANAGEMENT.S\s+DISCUSSION\s+AND\s+ANALYSIS\b",
        item + r"(?:7A\b|8\s*[.\-:]?\s+FINANCIAL\s+STATEMENTS\b)"
    )
    sections = []
    for title, body in [
        ("BUSINESS", business),
        ("RISK FACTORS", risks),
        ("MANAGEMENT DISCUSSION AND ANALYSIS", management)
    ]:
        if len(body) > 200:
            sections.append(f"{title}\n{body}")
    return "\n\n".join(sections)

def download_filings(ticker, years):
    ticker = ticker.strip().upper()
    years = int(years)
    if not re.fullmatch(r"[A-Z0-9.-]{1,10}", ticker):
        raise ValueError("Enter a valid stock ticker.")
    if not 1 <= years <= 5:
        raise ValueError("Select between 1 and 5 years.")

    downloader = Downloader("SECInsight", "9f06eshaseloth@gmail.com")
    downloader.get("10-K", ticker, limit=years)

    filing_dir = BASE_DIR / ticker / "10-K"
    if not filing_dir.exists():
        raise FileNotFoundError(f"SEC filing directory not found: {filing_dir.resolve()}")

    candidates = []
    for folder in filing_dir.iterdir():
        if not folder.is_dir():
            continue
        for file_path in folder.glob("*.txt"):
            try:
                raw = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            period = re.search(
                r"CONFORMED PERIOD OF REPORT:\s*(\d{8})",
                raw, re.I
            )
            filed = re.search(
                r"FILED AS OF DATE:\s*(\d{8})",
                raw, re.I
            )
            if not period:
                continue

            document = get_document(raw)
            cleaned = clean_text(document)
            sections = extract_sections(cleaned)

            candidates.append({
                "year": int(period.group(1)[:4]),
                "filed_date": filed.group(1) if filed else "",
                "content": cleaned,
                "analysis_content": sections,
                "source_file": str(file_path)
            })
            break

    candidates.sort(key=lambda x: x["filed_date"], reverse=True)
    selected = []
    seen = set()
    for filing in candidates:
        if filing["year"] not in seen:
            seen.add(filing["year"])
            selected.append(filing)
        if len(selected) >= years:
            break

    selected.sort(key=lambda x: x["year"])
    if not selected:
        raise RuntimeError("No 10-K filings with reporting-period metadata were found.")
    return selected
