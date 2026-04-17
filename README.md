"# cs517-assignment-2"

# 10-Q Filing Embeddings & LLM Query Pipeline

A three-step pipeline to download SEC 10-Q filings, parse and embed them into a FAISS index, and query them using an LLM.

## Step 1 — Download 10-Q Filings

Run the downloader script to fetch all the 10-Q filings needed to build the embeddings:

```bash
python download_10q.py
```

This will retrieve the relevant SEC 10-Q filings and save them locally for processing in the next step.

---

## Step 2 — Parse Filings & Build FAISS Index

Once the filings are downloaded, run the processing script to parse each filing and generate embeddings stored in a FAISS index:

```bash
python run.py
```

This script will:

- Parse all downloaded 10-Q filings
- Generate embeddings for each filing's content
- Store the embeddings in a local `faiss_index/` directory

---

## Step 3 — Query with the LLM

Use `llm.py` to query the embeddings. It supports two modes:

### File Mode

Provide a questions file and an output file as arguments:

```bash
python llm.py <questions.txt> <output_file>
```

**Example:**

```bash
python llm.py questions.txt answers.txt
```

- `<questions.txt>` — A plain text file with one question per line
- `<output_file>` — The file where answers will be written

### Interactive Mode

Run the script without arguments to enter interactive mode, where you can type questions directly:

```bash
python llm.py
```

You will be prompted to enter questions one at a time. Type `exit` or `quit` to end the session.

---

## Pipeline Overview

```
download_10q.py  →  run.py  →  llm.py
  (Download)       (Embed)     (Query)
```

| Step | Script            | Output                               |
| ---- | ----------------- | ------------------------------------ |
| 1    | `download_10q.py` | Raw 10-Q filing files                |
| 2    | `run.py`          | `faiss_index/` embeddings            |
| 3    | `llm.py`          | Answers via file or interactive mode |
