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
    # Fix "$ 8,268" → "$8,268"
    text = re.sub(r'\$\s+', '$', text)
    # Fix "7 %" → "7%"
    text = re.sub(r'(\d)\s+%', r'\1%', text)
    return text


def clean_row(cells: list[str]) -> str:
    cleaned = [c for c in cells if c.strip()]
    if not cleaned:
        return ""
    return "| " + " | ".join(cleaned) + " |"


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


def table_rows_to_nl_docs(rows: list[str], prefix: str, metadata: dict) -> list[Document]:
    """
    Convert period-labeled table rows into natural language sentences.
    Assumes rows[0] is a header like: | Metric | Q3 FY2025 | Q3 FY2024 |
    Each data row becomes its own Document so the embedding model can
    retrieve specific metrics semantically rather than scanning raw markdown.
    Example output:
      "[AAPL 10-Q 2025-08-01] Net sales: 94,930 (Q3 FY2025) vs 85,777 (Q3 FY2024)"
    """
    if not rows:
        return []

    # Parse period labels from header row
    header_cells = [c.strip() for c in rows[0].split('|') if c.strip()]
    periods = header_cells[1:]  # first cell is "Metric"

    docs = []
    for row in rows[1:]:
        if not row or '---' in row:
            continue
        cells = [c.strip() for c in row.split('|') if c.strip()]
        if len(cells) < 2:
            continue

        metric = cells[0]
        values = cells[1:]

        # Skip rows with no metric label or all-empty values
        if not metric or not any(v for v in values):
            continue

        parts = []
        for i, val in enumerate(values):
            period = periods[i] if i < len(periods) else f"col{i+1}"
            parts.append(f"{val} ({period})")

        sentence = f"{prefix} {metric}: {' vs '.join(parts)}"
        doc = Document(
            page_content=sentence,
            metadata={**metadata, 'content_type': 'table_nl'}
        )
        doc.metadata['section'] = tag_section(sentence)
        doc.metadata['note_number'] = tag_note_number(sentence)
        doc.metadata['period'] = periods[0] if periods else None
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
        rows = []
        header_injected = False
        pending_period_label = None  # 👈 remembers "Three Months Ended" across rows

        for tr in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            cleaned = clean_row(cells)
            if not cleaned:
                continue

            if not header_injected:
                # Step 1: detect "Three/Six/Nine Months Ended" row (no dates yet)
                if re.search(r'(three|six|nine)\s+months\s+ended', cleaned.lower()):
                    pending_period_label = re.search(
                        r'(three|six|nine)\s+months\s+ended', cleaned, re.IGNORECASE
                    ).group(0)
                    # Check if dates are also on this same row
                    dates = re.findall(
                        r'((?:january|february|march|april|may|june|july|august|'
                        r'september|october|november|december)\s+\d+,\s+\d{4})',
                        cleaned, flags=re.IGNORECASE
                    )
                    if len(dates) >= 2:
                        # Dates and label on same row — rewrite immediately
                        q1 = get_fiscal_quarter(dates[0], ticker) or dates[0]
                        q2 = get_fiscal_quarter(dates[1], ticker) or dates[1]
                        rows.append(f"| Metric | {q1} | {q2} |")
                        header_injected = True
                        pending_period_label = None
                    # else: dates are on the next row, skip this row entirely
                    continue

                # Step 2: if previous row was "Three Months Ended", this row has the dates
                if pending_period_label is not None:
                    dates = re.findall(
                        r'((?:january|february|march|april|may|june|july|august|'
                        r'september|october|november|december)\s+\d+,\s+\d{4})',
                        cleaned, flags=re.IGNORECASE
                    )
                    if len(dates) >= 2:
                        q1 = get_fiscal_quarter(dates[0], ticker) or dates[0]
                        q2 = get_fiscal_quarter(dates[1], ticker) or dates[1]
                        rows.append(f"| Metric | {q1} | {q2} |")
                        header_injected = True
                        pending_period_label = None
                        continue
                    elif len(dates) == 1:
                        q1 = get_fiscal_quarter(dates[0], ticker) or dates[0]
                        rows.append(f"| Metric | {q1} |")
                        header_injected = True
                        pending_period_label = None
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
        if len(text) > 80:
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
    if doc.metadata.get('content_type') == 'table':
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
