"""
SEC EDGAR 10-Q Downloader
Downloads 10-Q filings from SEC EDGAR for top 30 Nasdaq companies.
Prefers native PDFs; falls back to saving the primary HTML document as-is.

Requirements:
    pip install requests
"""

import re
import time
import requests
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "MyResearchLab admin@researchlab.com",
    "Accept-Encoding": "gzip, deflate",
}

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "AVGO", "TSLA", "GOOGL", "COST", "NFLX",
    "AMD",  "TMUS", "INTU", "QCOM", "AMAT", "ISRG", "TXN",  "HON",  "BKNG", "LRCX",
    "VRTX", "ADP",  "ADI",  "MU",   "PANW", "MELI", "REGN", "KLAC", "SBUX", "SNPS",
]

START_DATE = "2025-01-01"
END_DATE   = "2025-12-31"
OUTPUT_DIR = Path("sec-edgar-10q-filings")

REQUEST_DELAY = 0.5
DOWNLOAD_DELAY = 1.0


# ── CIK Lookup ────────────────────────────────────────────────────────────────

_TICKER_MAP: dict = {}

def load_ticker_map() -> dict:
    global _TICKER_MAP
    if _TICKER_MAP:
        return _TICKER_MAP
    print("Fetching SEC ticker->CIK map ...")
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=HEADERS, timeout=30
    )
    r.raise_for_status()
    for entry in r.json().values():
        _TICKER_MAP[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
    print(f"  Loaded {len(_TICKER_MAP)} tickers.\n")
    return _TICKER_MAP


def get_cik(ticker: str):
    return load_ticker_map().get(ticker.upper())


# ── Filing Discovery ──────────────────────────────────────────────────────────

def get_filings(cik: str, start: str, end: str) -> list:
    r = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=HEADERS, timeout=15
    )
    r.raise_for_status()
    recent     = r.json().get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    filings = []
    for form, date, acc in zip(forms, dates, accessions):
        if form == "10-Q" and start <= date <= end:
            filings.append({
                "date":          date,
                "accession":     acc.replace("-", ""),
                "accession_fmt": acc,
            })
    return filings


# ── Document Discovery ────────────────────────────────────────────────────────

def get_filing_documents(cik: str, accession_nodash: str, accession_fmt: str) -> dict:
    """
    Parse the filing index and return:
      { "pdf": <url or None>, "html": <url or None> }
    """
    cik_int   = int(cik)
    base      = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}"
    index_url = f"{base}/{accession_fmt}-index.htm"

    result = {"pdf": None, "html": None}

    r = requests.get(index_url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        r = requests.get(base + "/", headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return result

    html_text = r.text

    # Look for any PDF
    pdf_matches = re.findall(
        r'href="(/Archives/edgar/data/[^"]+\.pdf)"',
        html_text, re.IGNORECASE
    )
    if pdf_matches:
        result["pdf"] = "https://www.sec.gov" + pdf_matches[0]
    else:
        pdf_rel = re.findall(r'href="([^"/][^"]*\.pdf)"', html_text, re.IGNORECASE)
        if pdf_rel:
            result["pdf"] = f"{base}/{pdf_rel[0]}"

    # Look for primary HTML document (first .htm that isn't the index itself)
    htm_matches = re.findall(
        r'href="(/Archives/edgar/data/[^"]+\.htm)"',
        html_text, re.IGNORECASE
    )
    for href in htm_matches:
        if "-index" not in href.lower():
            result["html"] = "https://www.sec.gov" + href
            break

    if not result["html"]:
        htm_rel = re.findall(r'href="([^"/][^"]*\.htm)"', html_text, re.IGNORECASE)
        for href in htm_rel:
            if "-index" not in href.lower():
                result["html"] = f"{base}/{href}"
                break

    return result


# ── Download ──────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"    ⏭  Already exists: {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=HEADERS, timeout=60, stream=True)
    if r.status_code != 200:
        return False
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for ticker in TICKERS:
        print(f"\n{'─'*50}")
        print(f"📋  {ticker}")

        try:
            cik = get_cik(ticker)
        except Exception as e:
            print(f"    ✗  Error loading ticker map: {e}")
            summary.append((ticker, 0, str(e)))
            continue

        if not cik:
            print(f"    ✗  Ticker not found in SEC database")
            summary.append((ticker, 0, "CIK not found"))
            continue

        print(f"    CIK: {cik}")
        time.sleep(REQUEST_DELAY)

        try:
            filings = get_filings(cik, START_DATE, END_DATE)
        except Exception as e:
            print(f"    ✗  Error fetching filings: {e}")
            summary.append((ticker, 0, str(e)))
            continue

        if not filings:
            print(f"    ℹ  No 10-Q filings found in {START_DATE} - {END_DATE}")
            summary.append((ticker, 0, "No filings in range"))
            continue

        print(f"    Found {len(filings)} filing(s)")

        downloaded = 0
        for filing in filings:
            time.sleep(REQUEST_DELAY)
            try:
                docs = get_filing_documents(cik, filing["accession"], filing["accession_fmt"])
            except Exception as e:
                print(f"    ⚠  Error reading index for {filing['date']}: {e}")
                continue

            base_name = f"{ticker}_10Q_{filing['date']}_{filing['accession'][:8]}"
            ok = False

            # Prefer native PDF
            if docs["pdf"]:
                dest = OUTPUT_DIR / ticker / f"{base_name}.pdf"
                print(f"    ↓  {filing['date']}  [PDF]   ->  {dest.name}")
                time.sleep(DOWNLOAD_DELAY)
                try:
                    ok = download_file(docs["pdf"], dest)
                    if ok:
                        print(f"       ✓  Saved ({dest.stat().st_size // 1024} KB)")
                    else:
                        print(f"       ✗  Download failed")
                except Exception as e:
                    print(f"       ✗  {e}")

            # Fall back to raw HTML
            if not ok and docs["html"]:
                dest = OUTPUT_DIR / ticker / f"{base_name}.html"
                print(f"    ↓  {filing['date']}  [HTML]  ->  {dest.name}")
                time.sleep(DOWNLOAD_DELAY)
                try:
                    ok = download_file(docs["html"], dest)
                    if ok:
                        print(f"       ✓  Saved ({dest.stat().st_size // 1024} KB)")
                    else:
                        print(f"       ✗  Download failed")
                except Exception as e:
                    print(f"       ✗  {e}")

            if not docs["pdf"] and not docs["html"]:
                print(f"    ⚠  No documents found for {filing['date']} ({filing['accession_fmt']})")

            if ok:
                downloaded += 1

        summary.append((ticker, downloaded, "OK" if downloaded else "No files saved"))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    total = 0
    for ticker, count, status in summary:
        flag = "✓" if count > 0 else "✗"
        print(f"  {flag}  {ticker:<6}  {count} file(s)  [{status}]")
        total += count
    print(f"\n  Total files saved: {total}")
    print(f"  Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()