import re
import os
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from datetime import datetime


MAX_TABLE_CHARS = 10000

SECTION_PATTERNS = {
    "income_statement": r"(consolidated\s+)?(statements?\s+of\s+(operations|income|earnings|comprehensive\s+income))",
    "balance_sheet": r"(consolidated\s+)?(balance\s+sheets?|statements?\s+of\s+financial\s+position)",
    "cash_flow": r"(consolidated\s+)?(statements?\s+of\s+cash\s+flows?)",
    "stockholders_equity": r"(consolidated\s+)?(statements?\s+of\s+(stockholders|shareholders)['\s]+equity)",
    "note_summary_of_accounting": r"note\s*\d+\s*[–-]\s*(summary of significant accounting|basis of presentation)",
    "note_revenue": r"note\s*\d+\s*[–-]\s*revenue",
    "note_debt": r"note\s*\d+\s*[–-]\s*(debt|borrowings|credit facility)",
    "note_equity": r"note\s*\d+\s*[–-]\s*(equity|stock|share)",
    "note_income_taxes": r"note\s*\d+\s*[–-]\s*income tax",
    "note_earnings_per_share": r"note\s*\d+\s*[–-]\s*earnings per share",
    "note_segment": r"note\s*\d+\s*[–-]\s*segment",
    "note_acquisitions": r"note\s*\d+\s*[–-]\s*(acquisitions?|business combinations?)",
    "note_fair_value": r"note\s*\d+\s*[–-]\s*fair value",
    "note_commitments": r"note\s*\d+\s*[–-]\s*(commitments|contingencies|legal)",
    "note_leases": r"note\s*\d+\s*[–-]\s*leases?",
    "note_derivatives": r"note\s*\d+\s*[–-]\s*(derivatives?|hedging|financial instruments)",
    "note_restructuring": r"note\s*\d+\s*[–-]\s*restructuring",
    "note_goodwill": r"note\s*\d+\s*[–-]\s*(goodwill|intangible)",
    "note_inventory": r"note\s*\d+\s*[–-]\s*inventor",
    "mda_overview": r"(management'?s?\s+discussion|executive\s+overview|business\s+overview)",
    "mda_revenue": r"(net\s+sales|revenue\s+(discussion|analysis|overview))",
    "mda_gross_margin": r"gross\s+(margin|profit)\s+(discussion|analysis)?",
    "mda_operating_expenses": r"operating\s+expenses?\s+(discussion|analysis)?",
    "mda_rd_expense": r"(research\s+and\s+development|r&d)\s+(expense|discussion|analysis)?",
    "mda_liquidity": r"liquidity\s+and\s+capital\s+resources",
    "mda_critical_accounting": r"critical\s+accounting\s+(policies|estimates)",
    "mda_market_risk": r"quantitative\s+and\s+qualitative\s+disclosures\s+about\s+market\s+risk",
    "risk_factors": r"risk\s+factors",
    "legal_proceedings": r"legal\s+proceedings",
    "unresolved_comments": r"unresolved\s+staff\s+comments",
    "signature": r"(/s/|pursuant\s+to|chief\s+financial\s+officer|certif(y|ication)|sarbanes)",
    "table_of_contents": r"(table\s+of\s+contents|index\s+to\s+(financial|condensed))",
    "cover_page": r"(form\s+10-[qk]|securities\s+and\s+exchange\s+commission|commission\s+file\s+number)",
}

def tag_section(text: str) -> str:
    text_lower = text.lower()
    for section, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, text_lower):
            return section
    return "general"

def tag_note_number(text: str) -> str | None:
    match = re.search(r"note\s*(\d+)\s*[–-]", text.lower())
    return match.group(1) if match else None

NOISE_SECTIONS = {"signature", "table_of_contents", "cover_page"}

FINANCIAL_KEYWORDS = [
    'balance', 'sheet', 'income', 'cash', 'flows', 'amortization',
    'assets', 'liabilities', 'equity', 'operations', 'net sales',
    'revenue', 'expenses', 'earnings', 'gross margin', 'operating'
]

BOILERPLATE_PHRASES = [
    "pursuant to section 13 or 15",
    "indicate by check mark",
    "for such shorter period",
    "large accelerated filer",
    "state or other jurisdiction of incorporation",
    "address of principal executive offices",
    "pursuant to 18 u.s.c. section 1350",
    "sarbanes-oxley act of 2002",
    "certify, as of the date hereof",
    "not contain any untrue statement of a material fact",
    "fairly present in all material respects",
    "rule 13a-14(a) / 15d-14(a) certification",
]

TECHNICAL_NOISE_KEYWORDS = [
    'xbrli:', 'namespace prefix:', 'period type:', 'balance type:',
    'data type:', 'definition available', 'http://fasb.org/',
    'http://www.xbrl.org/', 'abstract namespace'
]


def is_relevant_table(table) -> bool:
    """Keep only tables that contain financial content; drop XBRL taxonomy tables."""
    table_text = table.get_text(" ", strip=True).lower()
    taxonomy_indicators = ['namespace prefix', 'data type', 'balance type']
    if any(ind in table_text for ind in taxonomy_indicators):
        return False
    return any(k in table_text for k in FINANCIAL_KEYWORDS)


def is_technical_noise(text: str) -> bool:
    """Return True for XBRL namespace walls and taxonomy definition blocks."""
    text_lower = text.lower()
    if "x - definition" in text_lower or "x - references" in text_lower:
        return True
    if any(kw in text_lower for kw in TECHNICAL_NOISE_KEYWORDS):
        return True
    xbrl_patterns = [r'us-gaap:', r'iso4217:', r'xbrli:', r'\d{10}']
    return sum(1 for p in xbrl_patterns if re.search(p, text)) >= 2


def is_boilerplate(text: str) -> bool:
    """Return True for SOX certifications and cover-page boilerplate."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in BOILERPLATE_PHRASES)


# ─────────────────────────────────────────────
# FISCAL CALENDAR MAPPING
# ─────────────────────────────────────────────

FISCAL_YEAR_END_MONTH = {
    "AAPL": 9,   # September
    "MSFT": 6,   # June
    "NVDA": 1,   # January
    "AMZN": 12,  # December
    "META": 12,  # December
    "AVGO": 10,  # October/Early November
    "TSLA": 12,  # December
    "GOOGL": 12, # December
    "COST": 8,   # August
    "NFLX": 12,  # December
    "AMD": 12,   # December
    "TMUS": 12,  # December
    "INTU": 7,   # July
    "QCOM": 9,   # September
    "AMAT": 10,  # October
    "ISRG": 12,  # December
    "TXN": 12,   # December
    "HON": 12,   # December
    "BKNG": 12,  # December
    "LRCX": 6,   # June
    "VRTX": 12,  # December
    "ADP": 6,    # June
    "ADI": 10,   # October
    "MU": 8,     # August
    "PANW": 7,   # July
    "MELI": 12,  # December
    "REGN": 12,  # December
    "KLAC": 6,   # June
    "SBUX": 9,   # September
    "SNPS": 10,  # October
}

def get_fiscal_quarter(date_str: str, ticker: str) -> str | None:
    """
    Convert a date string like 'December 28, 2024' into a fiscal quarter
    label like 'Q1 FY2025', using the company's fiscal year end month.
    """
    try:
        date = datetime.strptime(date_str.strip(), "%B %d, %Y")
    except ValueError:
        return None

    fy_end_month = FISCAL_YEAR_END_MONTH.get(ticker.upper(), 12)

    # Shift so fiscal year end month = month 12, then derive quarter
    shifted = (date.month - fy_end_month - 1) % 12
    quarter = (shifted // 3) + 1

    # Fiscal year: if past fiscal year end month, FY = calendar year + 1
    if date.month > fy_end_month:
        fiscal_year = date.year + 1
    else:
        fiscal_year = date.year

    return f"Q{quarter} FY{fiscal_year}"


def extract_fiscal_period(text: str, ticker: str = "") -> str | None:
    """Extract first date found and convert to fiscal quarter label."""
    match = re.search(
        r'((?:january|february|march|april|may|june|july|august|'
        r'september|october|november|december)\s+\d+,\s+\d{4})',
        text,
        flags=re.IGNORECASE
    )
    if match:
        return get_fiscal_quarter(match.group(1), ticker)
    return None


# ─────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────

def clean_financial_chunk(text: str) -> str:
    text = text.replace('\xa0', ' ')          # non-breaking space -> regular space
    text = re.sub(r'\$\s+', '$', text)        # "$ 8,268" -> "$8,268"
    text = re.sub(r'(\d)\s+%', r'\1%', text)  # "7 %" -> "7%"
    text = re.sub(r'\s+', ' ', text)          # collapse multiple spaces
    return text.strip()


def clean_row(cells: list[str]) -> str:
    """Build a pipe-delimited markdown row, merging lone '$' or '(' prefixes
    with the following cell (e.g. ['$', '8,268'] -> '$8,268')."""
    merged = []
    skip_next = False
    for i, cell in enumerate(cells):
        if skip_next:
            skip_next = False
            continue
        c = cell.strip()
        if c in ('$', '(', '$(') and i + 1 < len(cells):
            merged.append(f"{c}{cells[i + 1].strip()}")
            skip_next = True
        else:
            merged.append(c)
    final = [c for c in merged if c]
    if not final:
        return ""
    return "| " + " | ".join(final) + " |"


def rewrite_period_header(cleaned: str, ticker: str = "") -> str | None:
    """
    If this row is a 'Three/Six/Nine Months Ended ...' subheader,
    rewrite it as a proper markdown column header with fiscal quarter labels.
    Returns None if the row is not a period header.
    """
    if not re.search(r'(three|six|nine)\s+months\s+ended', cleaned.lower()):
        return None

    dates = re.findall(
        r'((?:january|february|march|april|may|june|july|august|'
        r'september|october|november|december)\s+\d+,\s+\d{4})',
        cleaned,
        flags=re.IGNORECASE
    )

    if len(dates) >= 2:
        q1 = get_fiscal_quarter(dates[0], ticker) or dates[0]
        q2 = get_fiscal_quarter(dates[1], ticker) or dates[1]
        return f"| Metric | {q1} | {q2} |"
    elif len(dates) == 1:
        q1 = get_fiscal_quarter(dates[0], ticker) or dates[0]
        return f"| Metric | {q1} |"

    return cleaned


def split_large_table(rows: list[str], prefix: str, metadata: dict, max_chars: int) -> list[Document]:
    header = rows[0]
    col_count = header.count("|") - 1
    separator = "| " + " | ".join(["---"] * col_count) + " |"

    sub_chunks = []
    current = [header, separator]
    current_len = len(header) + len(separator)

    for row in rows[1:]:
        if not row:
            continue
        if current_len + len(row) > max_chars:
            sub_chunks.append('\n'.join(current))
            current = [header, separator, row]
            current_len = len(header) + len(separator) + len(row)
        else:
            current.append(row)
            current_len += len(row)

    if len(current) > 2:
        sub_chunks.append('\n'.join(current))

    return [
        Document(
            page_content=f"{prefix}\n{sub}",
            metadata={**metadata, 'content_type': 'table'}
        )
        for sub in sub_chunks
    ]


def _build_period_header(period_types: list[str], dates: list[str], ticker: str) -> str:
    """Build a pipe-delimited header row with rich period labels.
    If 2+ period types (e.g. Three + Nine months) and 4+ dates: 4 columns.
    Otherwise: 2 columns using the first period type.
    Column format: '{period_type} {calendar_date} ({fiscal_quarter})'
    """
    cols = []
    if len(period_types) >= 2 and len(dates) >= 4:
        for i, d in enumerate(dates[:4]):
            pt = period_types[0] if i < 2 else period_types[1]
            q = get_fiscal_quarter(d, ticker) or d
            cols.append(f"{pt} {d} ({q})")
    else:
        pt = period_types[0] if period_types else "three months ended"
        for d in dates[:2]:
            q = get_fiscal_quarter(d, ticker) or d
            cols.append(f"{pt} {d} ({q})")
    return "| Metric | " + " | ".join(cols) + " |"


def table_rows_to_nl_docs(rows: list[str], prefix: str, metadata: dict) -> list[Document]:
    """
    Convert period-labeled table rows into one Document per (metric, period) pair.
    Header format: | Metric | three months ended June 28, 2025 (Q3 FY2025) | ... |
    Output: "[AAPL 10-Q 2025-08-01] Net sales for three months ended June 28, 2025 (Q3 FY2025): 94,036"
    Columns with no header label are skipped (prevents YTD/quarterly mixing).
    """
    if not rows:
        return []

    header_cells = [c.strip() for c in rows[0].split('|') if c.strip()]
    periods = header_cells[1:]  # e.g. ["three months ended June 28, 2025 (Q3 FY2025)", ...]

    docs = []
    for row in rows[1:]:
        if not row or '---' in row:
            continue
        cells = [c.strip() for c in row.split('|') if c.strip()]
        if len(cells) < 2:
            continue

        metric = cells[0]
        values = cells[1:]

        if not metric or not any(v for v in values):
            continue

        # One doc per (metric, period) pair — no mixing of periods in one sentence
        for i, val in enumerate(values):
            if not val or val in ('-', '\u2014', '\u2013'):
                continue
            if i >= len(periods):
                continue  # skip unlabeled extra columns (e.g. YTD when header only covers quarterly)
            period = periods[i]
            sentence = f"{prefix} {metric} for {period}: {val}"
            doc = Document(
                page_content=sentence,
                metadata={**metadata, 'content_type': 'table_nl'}
            )
            doc.metadata['section'] = tag_section(sentence)
            doc.metadata['note_number'] = tag_note_number(sentence)
            doc.metadata['period'] = period
            docs.append(doc)

    return docs


def parse_sec_html(filepath: str, ticker: str, filing_date: str) -> list[Document]:
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    documents = []
    prefix = f"[{ticker} 10-Q {filing_date}]"
    metadata = {
        'ticker': ticker,
        'filing_date': filing_date,
        'source': filepath,
        'form': '10-Q'
    }

    # --- Extract tables ---
    for table in soup.find_all('table'):
        if not is_relevant_table(table):
            continue

        rows = []
        header_injected = False
        pending_period_types = []  # captures all matched period strings across rows

        DATE_PAT = (
            r'((?:january|february|march|april|may|june|july|august|'
            r'september|october|november|december)\s+\d+,\s+\d{4})'
        )

        for tr in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            cleaned = clean_row(cells)
            if not cleaned:
                continue

            if not header_injected:
                # Step 1: detect "Three/Six/Nine Months Ended" row
                period_matches = re.findall(
                    r'(?:three|six|nine)\s+months\s+ended', cleaned, re.IGNORECASE
                )
                if period_matches:
                    pending_period_types = period_matches
                    dates = re.findall(DATE_PAT, cleaned, flags=re.IGNORECASE)
                    if len(dates) >= 2:
                        # Dates and period labels on same row — build header immediately
                        rows.append(_build_period_header(pending_period_types, dates, ticker))
                        header_injected = True
                        pending_period_types = []
                    # else: dates are on the next row, skip this row entirely
                    continue

                # Step 2: previous row was the period-type row; this row has the dates
                if pending_period_types:
                    dates = re.findall(DATE_PAT, cleaned, flags=re.IGNORECASE)
                    if len(dates) >= 2:
                        rows.append(_build_period_header(pending_period_types, dates, ticker))
                        header_injected = True
                        pending_period_types = []
                        continue
                    elif len(dates) == 1:
                        q1 = get_fiscal_quarter(dates[0], ticker) or dates[0]
                        col = f"{pending_period_types[0]} {dates[0]} ({q1})"
                        rows.append(f"| Metric | {col} |")
                        header_injected = True
                        pending_period_types = []
                        continue

            rows.append(clean_financial_chunk(cleaned))

        if len(rows) < 2:
            continue

        if header_injected:
            # Period header was detected — convert rows to NL sentences for
            # better semantic retrieval of specific financial metrics
            nl_docs = table_rows_to_nl_docs(rows, prefix, metadata)
            documents.extend(nl_docs)
        else:
            # No period header — fall back to storing as markdown table
            table_text = '\n'.join(rows)
            if len(table_text) > MAX_TABLE_CHARS:
                docs = split_large_table(rows, prefix, metadata, MAX_TABLE_CHARS)
                for doc in docs:
                    doc.metadata["section"] = tag_section(doc.page_content)
                    doc.metadata["note_number"] = tag_note_number(doc.page_content)
                    doc.metadata["period"] = extract_fiscal_period(doc.page_content, ticker)
                documents.extend(docs)
            else:
                doc = Document(
                    page_content=f"{prefix}\n{table_text}",
                    metadata={**metadata, 'content_type': 'table'}
                )
                doc.metadata["section"] = tag_section(doc.page_content)
                doc.metadata["note_number"] = tag_note_number(doc.page_content)
                doc.metadata["period"] = extract_fiscal_period(table_text, ticker)
                documents.append(doc)

    # --- Extract narrative text ---
    for tag in soup.find_all(['p', 'div', 'li']):
        if tag.find_parent('table'):
            continue
        text = clean_financial_chunk(
            tag.get_text(separator=' ', strip=True)
        )
        if len(text) > 80 and not is_technical_noise(text) and not is_boilerplate(text):
            doc = Document(
                page_content=f"{prefix}\n{text}",
                metadata={**metadata, 'content_type': 'text'}
            )
            doc.metadata["section"] = tag_section(doc.page_content)
            doc.metadata["note_number"] = tag_note_number(doc.page_content)
            doc.metadata["period"] = extract_fiscal_period(text, ticker)
            if doc.metadata["section"] not in NOISE_SECTIONS:
                documents.append(doc)

    return documents


def load_all_filings(base_dir: str = './sec-edgar-10q-filings') -> list[Document]:
    all_docs = []
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith('.html'):
                continue
            path = os.path.join(root, fname)
            # Structure: base_dir/{TICKER}/{filename}
            # Extract ticker from the immediate parent folder name
            ticker = os.path.basename(os.path.dirname(path))
            # Extract date from filename: TICKER_10Q_YYYY-MM-DD_...html
            date_match = re.search(r'_(\d{4}-\d{2}-\d{2})_', fname)
            filing_date = date_match.group(1) if date_match else 'Unknown'

            docs = parse_sec_html(path, ticker, filing_date)
            all_docs.extend(docs)
    return all_docs


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
print("Loading and parsing HTML filings...")
docs = load_all_filings('./sec-edgar-10q-filings')

table_docs   = [d for d in docs if d.metadata.get('content_type') == 'table']
table_nl_docs = [d for d in docs if d.metadata.get('content_type') == 'table_nl']
text_docs    = [d for d in docs if d.metadata.get('content_type') == 'text']
print(f"Loaded {len(docs)} total sections")
print(f"  -> {len(table_nl_docs)} NL table rows (period-labeled)")
print(f"  -> {len(table_docs)} markdown table blocks (no period header)")
print(f"  -> {len(text_docs)} narrative blocks")

if table_docs:
    print("\nSample table block:")
    print(table_docs[0].page_content[:600])
    print()


# ─────────────────────────────────────────────
# SPLIT — only narrative text, never tables
# ─────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n\n", "\n\n", "\n", " ", ""],
    keep_separator=True
)

final_chunks = []
for doc in docs:
    if doc.metadata.get('content_type') in ('table', 'table_nl'):
        final_chunks.append(doc)
    else:
        splits = splitter.split_documents([doc])
        for chunk in splits:
            chunk.metadata["section"] = tag_section(chunk.page_content)
            chunk.metadata["note_number"] = tag_note_number(chunk.page_content)
            chunk.metadata["period"] = extract_fiscal_period(chunk.page_content)
            if chunk.metadata["section"] not in NOISE_SECTIONS:
                final_chunks.append(chunk)

print(f"Final chunk count: {len(final_chunks)}")

truncated = [
    c for c in final_chunks
    if c.metadata.get('content_type') == 'table'
    and c.page_content.rstrip().endswith('|')
]
print(f"Truncated table chunks: {len(truncated)}")


# ─────────────────────────────────────────────
# EMBED + SAVE
# ─────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"},  # use "mps" on Apple Silicon
    encode_kwargs={
        "batch_size": 32,
        "normalize_embeddings": True
    }
)

vectorstore = FAISS.from_documents(final_chunks, embeddings, distance_strategy=DistanceStrategy.COSINE)
vectorstore.save_local("faiss_index")
print("\nVectorstore saved to 'faiss_index'")
