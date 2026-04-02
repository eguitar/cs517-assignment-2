import os
import re

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
MAX_TABLE_CHARS = 4000

FINANCIAL_KEYWORDS = [
    'balance', 'sheet', 'income', 'cash', 'flows',
    'amortization', 'assets', 'liabilities', 'equity',
    'operations', 'net sales', 'revenue', 'expenses'
]

SECTION_PATTERNS = {
    "income_statement": r"(statements?\s+of\s+(operations|income|earnings))",
    "balance_sheet": r"(balance\s+sheets?)",
    "cash_flow": r"(cash\s+flows?)",
    "mda": r"(management'?s?\s+discussion)",
    "risk_factors": r"risk\s+factors",
    "note": r"note\s*\d+",
    "general": r".*"
}

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
    "officer’s certificate of the registrant"
]

TECHNICAL_NOISE_KEYWORDS = [
    'xbrli:', 'namespace prefix:', 'period type:', 'balance type:',
    'data type:', 'definition available', 'http://fasb.org/',
    'http://www.xbrl.org/', 'abstract namespace'
]

NOISE_SECTIONS = {"signature", "table_of_contents", "cover_page"}


# =========================
# PERIOD EXTRACTION
# =========================
def extract_period(content):
    match = re.search(
        r'<dei:DocumentPeriodEndDate[^>]*contextRef="([^"]+)"[^>]*>(.*?)</dei:DocumentPeriodEndDate>',
        content,
        re.IGNORECASE
    )
    if not match:
        return "UNKNOWN_PERIOD"

    context_ref = match.group(1)

    fy = re.search(
        rf'<dei:DocumentFiscalYearFocus[^>]*contextRef="{context_ref}"[^>]*>(.*?)</dei:DocumentFiscalYearFocus>',
        content,
        re.IGNORECASE
    )
    fq = re.search(
        rf'<dei:DocumentFiscalPeriodFocus[^>]*contextRef="{context_ref}"[^>]*>(.*?)</dei:DocumentFiscalPeriodFocus>',
        content,
        re.IGNORECASE
    )

    if fy and fq:
        return f"{fy.group(1)}-{fq.group(1).replace('QTR','Q')}"

    return "UNKNOWN_PERIOD"


# =========================
# TABLE PROCESSING
# =========================
def clean_text(text):
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_relevant_table(table):
    table_text = table.get_text(" ", strip=True).lower()
    taxonomy_indicators = ['namespace prefix', 'data type', 'balance type']
    if any(ind in table_text for ind in taxonomy_indicators):
        return False
    return any(k in table_text for k in FINANCIAL_KEYWORDS)


def table_to_markdown(table):
    rows = []
    for tr in table.find_all('tr'):
        cells = [clean_text(td.get_text(strip=True)) for td in tr.find_all(['td', 'th'])]

        merged_cells = []
        skip_next = False
        for i in range(len(cells)):
            if skip_next:
                skip_next = False
                continue

            current = cells[i]
            if current in ['$', '(', '$ ('] and i + 1 < len(cells):
                merged_cells.append(f"{current}{cells[i + 1]}")
                skip_next = True
            else:
                merged_cells.append(current)

        final_cells = [c for c in merged_cells if c]

        if final_cells:
            rows.append("| " + " | ".join(final_cells) + " |")

    return rows


def split_table(rows, metadata):
    header = rows[0]
    separator = "| " + " | ".join(["---"] * (header.count("|") - 1)) + " |"

    chunks, current = [], [header, separator]
    length = len(header)

    for row in rows[1:]:
        if length + len(row) > MAX_TABLE_CHARS:
            chunks.append("\n".join(current))
            current = [header, separator, row]
            length = len(row)
        else:
            current.append(row)
            length += len(row)

    chunks.append("\n".join(current))

    return [
        Document(page_content=c, metadata={**metadata, "content_type": "table"})
        for c in chunks
    ]


# =========================
# SECTION TAGGING
# =========================
def tag_section(text):
    text = text.lower()
    for name, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, text):
            return name
    return "general"


def tag_note(text):
    m = re.search(r"note\s*(\d+)", text.lower())
    return m.group(1) if m else None


def is_technical_noise(text):
    text_lower = text.lower()
    if "x - definition" in text_lower or "x - references" in text_lower:
        return True

    if any(keyword in text_lower for keyword in TECHNICAL_NOISE_KEYWORDS):
        return True

    xbrl_patterns = [r'us-gaap:', r'iso4217:', r'xbrli:', r'\d{10}']
    matches = sum(1 for p in xbrl_patterns if re.search(p, text))
    return matches >= 2

def is_boilerplate(text):
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in BOILERPLATE_PHRASES)


def extract_readable_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        is_readable_section = False
        buffer = []

        for line in f:
            if line.startswith('<DOCUMENT>'):
                is_readable_section = True

            if re.search(r'<(TYPE|FILENAME)>(GRAPHIC|PDF|ZIP|JPG)', line):
                is_readable_section = False

            if line.startswith('</DOCUMENT>'):
                is_readable_section = False

            if is_readable_section:
                if not (line.startswith('M') and len(line.strip()) == 61):
                    buffer.append(line)

        return "".join(buffer)

# =========================
# MAIN PARSER
# =========================
def process_filing(file_path, ticker):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    content = extract_readable_text(file_path)
    period = extract_period(content)
    soup = BeautifulSoup(content, 'lxml')
    documents = []

    base_metadata = {
        "ticker": ticker,
        "period": period,
        "source": file_path
    }

    # ---- TABLES ----
    for table in soup.find_all('table'):
        if not is_relevant_table(table):
            continue

        rows = table_to_markdown(table)
        if len(rows) < 2:
            continue

        docs = split_table(rows, base_metadata)
        for d in docs:
            d.metadata["section"] = tag_section(d.page_content)
            d.metadata["note"] = tag_note(d.page_content)
        documents.extend(docs)

    # ---- TEXT ----
    for tag in soup.find_all(['p', 'div', 'li']):
        if tag.find_parent('table'):
            continue

        text = clean_text(tag.get_text(separator=" ", strip=True))

        # New Filter: Skip short noise and technical XBRL walls
        if len(text) < 80 or is_technical_noise(text) or is_boilerplate(text):
            continue

        doc = Document(
            page_content=text,
            metadata={**base_metadata, "content_type": "text"}
        )

        doc.metadata["section"] = tag_section(text)
        doc.metadata["note"] = tag_note(text)

        if doc.metadata["section"] not in NOISE_SECTIONS:
            documents.append(doc)

    return documents


# =========================
# LOAD ALL FILES
# =========================
def load_filings(base_dir):
    all_docs = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file != "full-submission.txt":
                continue

            path = os.path.join(root, file)

            rel = os.path.relpath(path, base_dir).split(os.sep)
            ticker = rel[0]

            try:
                docs = process_filing(path, ticker)
                all_docs.extend(docs)
                print(f"Processed {path}")
            except Exception as e:
                print(f"Failed {path}: {e}")

    return all_docs


# =========================
# CHUNK TEXT ONLY
# =========================
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    final = []

    for doc in docs:
        if doc.metadata["content_type"] == "table":
            final.append(doc)
        else:
            splits = splitter.split_documents([doc])
            for s in splits:
                s.metadata["section"] = tag_section(s.page_content)
                s.metadata["note"] = tag_note(s.page_content)
                final.append(s)

    return final


def prioritized_search(vectorstore, query, k=4, table_boost=1.5):
    raw_results = vectorstore.similarity_search_with_relevance_scores(query, k=k * 2)

    scored_docs = []
    for doc, score in raw_results:
        final_score = score
        if doc.metadata.get("content_type") == "table":
            final_score = score * table_boost

        scored_docs.append((doc, final_score))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:k]]


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    base_dir = "sec-edgar-filings\\AAPL"

    docs = load_filings(base_dir)
    print(f"Loaded {len(docs)} raw docs")

    final_chunks = chunk_documents(docs)

    #build vector store
    if final_chunks:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        vectorstore = FAISS.from_documents(
            final_chunks,
            embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )
        query = "AAPL research and development expenses for 2025 Q2"
        retrieved_docs = prioritized_search(vectorstore, query, 4)

        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- Result {i + 1} ({doc.metadata}) ---")
            print(doc.page_content)


'''        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "batch_size": 64,
                "normalize_embeddings": True,
            }
        )

        vectorstore = FAISS.from_documents(
            final_chunks,
            embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )
        vectorstore.save_local("faiss_index")
        print("\nVectorstore saved to 'faiss_index'")
        '''