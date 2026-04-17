import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import re
import sys

# PROMPT 1
prompt = PromptTemplate.from_template("""<|system|>
You are a financial data extraction assistant specialized in SEC filings.

Response Guidelines:
- Answer ONLY using information explicitly stated in the provided 10-Q context
- If the data is not present, state: "This information is not available in the provided 10-Q"
- Report exact figures, dates, and values as they appear in the filing
- Use the same units and notation from the source (e.g., "in thousands", "in millions")
- Do not interpret, analyze, or infer beyond what is directly stated
- Do not repeat the question or instructions<|end|>

<|user|>
Context:
<context>
{context}
</context>

Question: {question}<|end|>

<|assistant|>
""")

# PROMPT 2
# prompt = PromptTemplate.from_template("""<|system|>
# You are a financial analyst assistant specialized in SEC 10-Q filings.

# Extraction rules:
# - Use ONLY information explicitly stated in the context chunks below
# - Preserve exact figures, units, and notation as they appear (e.g. "in millions", "$4.2B")
# - If a figure appears in multiple chunks with different values, report all of them and note the discrepancy
# - If the answer is not present, respond: "Not found in the provided context"
# - Do not calculate, infer, or extrapolate beyond what is directly stated
# - For YoY comparisons, only report them if both periods appear explicitly in the context

# Output format:
# 1. Direct answer (1–2 sentences with exact figures)

# <|user|>
# Context chunks:
# {context}

# Question: {question}<|end|>

# <|assistant|>
# """)

# PROMPT 3
# prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:

# {context}

# Question: {question}
# """)

model_id = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="cuda",
    # trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer,
    max_new_tokens=512, temperature=0.1, do_sample=True,
    return_full_text=False
)
llm = HuggingFacePipeline(pipeline=pipe)

# Must change to match embedding model
# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-base-en-v1.5",
#     model_kwargs={"device": "cuda"}
# )

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"}
)


vectorstore = FAISS.load_local(
    "./faiss_index", embeddings,
    allow_dangerous_deserialization=True
)

# ------------------ NOT USED ------------------ 
TICKER_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "meta": "META",
    "broadcom": "AVGO",
    "tesla": "TSLA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "costco": "COST",
    "netflix": "NFLX",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "t-mobile": "TMUS",
    "intuit": "INTU",
    "qualcomm": "QCOM",
    "applied materials": "AMAT",
    "intuitive surgical": "ISRG",
    "texas instruments": "TXN",
    "honeywell": "HON",
    "booking holdings": "BKNG",
    "booking": "BKNG",
    "lam research": "LRCX",
    "vertex pharmaceuticals": "VRTX",
    "vertex": "VRTX",
    "adp": "ADP",
    "automatic data processing": "ADP",
    "analog devices": "ADI",
    "micron": "MU",
    "micron technology": "MU",
    "palo alto networks": "PANW",
    "palo alto": "PANW",
    "mercadolibre": "MELI",
    "regeneron": "REGN",
    "regeneron pharmaceuticals": "REGN",
    "kla": "KLAC",
    "kla corporation": "KLAC",
    "starbucks": "SBUX",
    "synopsys": "SNPS",
}

def get_ticker_filter(query: str) -> dict | None:
    query_lower = query.lower()
    for name, ticker in TICKER_MAP.items():
        if name in query_lower or ticker.lower() in query_lower:
            return {"ticker": ticker}
    return None

def get_retriever(query: str):
    ticker_filter = get_ticker_filter(query)
    search_kwargs = {"k": 4}
    if ticker_filter:
        search_kwargs["filter"] = ticker_filter
    return vectorstore.as_retriever(search_kwargs=search_kwargs)

# TICKER prefiltering
def get_dynamic_retriever(query: str):
    ticker_filter = get_ticker_filter(query)
    search_kwargs = {"k": 4}
    if ticker_filter:
        search_kwargs["filter"] = ticker_filter
    return vectorstore.as_retriever(search_kwargs=search_kwargs)
# ------------------ NOT USED ------------------ 

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class CleanOutputParser(StrOutputParser):
    def parse(self, text: str) -> str:
        # Strip Phi-3 thinking tokens
        text = re.sub(r'<\|thinking\|>.*?<\|/thinking\|>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Strip any other special Phi-3 tokens
        text = re.sub(r'<\|.*?\|>', '', text)
        return text.strip()
    
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

# Defines the retriever
retriever = vectorstore.as_retriever( search_kwargs={"k": 5})

prioritized_retriever = RunnableLambda(
    lambda query: prioritized_search(vectorstore, query, k=10)
)

# Runs the full LLM chain 
# 1. Calls the retriever
# 2. passes the context from the retriever to LLM
# 3. LLM output is cleaned and outputted

chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
        source_documents=prioritized_retriever  # ✅ now a Runnable, not a static call
    )
    .assign(
        answer=prompt | llm | CleanOutputParser()
    )
)

def run_batch(questions_path: str, output_path: str):
    """Read questions one per line, write one answer per line to output."""
    with open(questions_path, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"Running batch on {len(questions)} questions → {output_path}")
    with open(output_path, 'w') as out:
        for i, question in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {question}")
            result = chain.invoke(question)
            answer = result['answer'].strip().replace('\n', ' ')
            out.write(answer + '\n')
    print("Done.")


def run_interactive():
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit"]:
            break

        print("Thinking...\n")
        result = chain.invoke(question)
        print("----- WITH RAG -----")
        print(f"Answer: {result['answer']}\n")

        print("----- RETRIEVED CHUNKS -----")
        for i, doc in enumerate(result['source_documents']):
            print(f"Chunk {i+1}:")
            print(f"Content: {doc.page_content[:300]}...")
            if doc.metadata:
                print(f"Metadata: {doc.metadata}")
            print("-" * 10)


# Run batch if arguments provided, otherwise interactive
# Usage: python llm.py questions.txt system_output.txt
if len(sys.argv) == 3:
    run_batch(sys.argv[1], sys.argv[2])
else:
    run_interactive()