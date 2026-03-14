from sec_edgar_downloader import Downloader
import os
import re

def download_filings(ticker, years):
    dl = Downloader("SECInsight", "test@example.com")
    dl.get("10-K", ticker, limit=years)
    
    filing_dir = f"sec-edgar-filings/{ticker}/10-K"
    filings = []
    
    for folder in sorted(os.listdir(filing_dir)):
        folder_path = os.path.join(filing_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    
                    year_match = re.search(r'CONFORMED PERIOD OF REPORT:\s*(\d{4})', content)
                    year = year_match.group(1) if year_match else folder[:4]
                    
                    clean = re.sub(r'<[^>]+>', ' ', content)
                    clean = re.sub(r'\s+', ' ', clean).strip()
                    
                    if len(clean) > 100:
                        filings.append({
                            "year": year,
                            "content": clean[:50000]
                        })
                        break

    return filings 