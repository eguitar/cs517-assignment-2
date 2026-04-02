from sec_edgar_downloader import Downloader

# 1. Initialize the downloader
# The SEC requires a User-Agent string: "Company Name AdminContact@domain.com"
dl = Downloader("MyResearchLab", "admin@researchlab.com")

# 2. List of top 30 Nasdaq tickers
tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "AVGO", "TSLA", "GOOGL", "COST", "NFLX",
    "AMD", "TMUS", "INTU", "QCOM", "AMAT", "ISRG", "TXN", "HON", "BKNG", "LRCX",
    "VRTX", "ADP", "ADI", "MU", "PANW", "MELI", "REGN", "KLAC", "SBUX", "SNPS"
]

# 3. Define the time range for 2025 filings
# Standard 10-Q filings for 2025 usually fall between 2025-01-01 and 2025-12-31
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

print(f"Starting download of 10-Q forms for {len(tickers)} companies...")

for ticker in tickers:
    try:
        # Download all 10-Q filings for the ticker in 2025
        num_downloaded = dl.get("10-Q", ticker, after=START_DATE, before=END_DATE)
        print(f"Successfully downloaded {num_downloaded} filings for {ticker}")
    except Exception as e:
        print(f"Could not download filings for {ticker}: {e}")

print("\nTask Complete. Filings are saved in the 'sec-edgar-filings' folder.")