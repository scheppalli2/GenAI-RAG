# GenAIRAG

RAG app using:
- OpenAI embeddings + OpenAI chat model
- ChromaDB (vector DB)
- Gradio UI

## 1) Activate env
```bash
cd /Users/sureshcheppalli/Projects/GenAIRAG
source .venv/bin/activate
```

## 2) Ensure OpenAI key exists
`rag_app.py` loads API key from `../MITWork/.env` (`OPENAI_API_KEY` or `OPENAI_KEY`).

## 3) Add documents
Put `.txt` files under `data/`.

## 4) Ingest docs
```bash
cd /Users/sureshcheppalli/Projects/GenAIRAG
COLLECTION=agentic_pdf_docs_openai .venv/bin/python rag_app.py ingest --folder ./data
```

## 5) Ask questions (CLI)
```bash
cd /Users/sureshcheppalli/Projects/GenAIRAG
COLLECTION=agentic_pdf_docs_openai .venv/bin/python rag_app.py ask --question "What are the key points?" --k 10
```

## 6) Run Gradio UI
```bash
cd /Users/sureshcheppalli/Projects/GenAIRAG
.venv/bin/python gradio_app.py
```
Open: `http://127.0.0.1:7860`
