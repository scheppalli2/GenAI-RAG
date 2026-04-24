# GenAIRAG

Minimal local RAG starter using:
- ChromaDB (vector DB)
- Ollama (embeddings + local SLM)

## 1) Activate env
```bash
cd /Users/sureshcheppalli/Projects/GenAIRAG
source .venv/bin/activate
```

## 2) Start Ollama (project-local)
```bash
HOME=/Users/sureshcheppalli/Projects/GenAIRAG/.home \
OLLAMA_MODELS=/Users/sureshcheppalli/Projects/GenAIRAG/.ollama/models \
/Users/sureshcheppalli/Projects/GenAIRAG/tools/ollama serve
```

## 3) Pull models (once)
```bash
HOME=/Users/sureshcheppalli/Projects/GenAIRAG/.home \
OLLAMA_MODELS=/Users/sureshcheppalli/Projects/GenAIRAG/.ollama/models \
OLLAMA_HOST=http://127.0.0.1:11434 \
/Users/sureshcheppalli/Projects/GenAIRAG/tools/ollama pull nomic-embed-text

HOME=/Users/sureshcheppalli/Projects/GenAIRAG/.home \
OLLAMA_MODELS=/Users/sureshcheppalli/Projects/GenAIRAG/.ollama/models \
OLLAMA_HOST=http://127.0.0.1:11434 \
/Users/sureshcheppalli/Projects/GenAIRAG/tools/ollama pull qwen2.5:3b
```

## 4) Add documents
Put `.txt` files under `data/`.

## 5) Ingest docs
```bash
cd /Users/sureshcheppalli/Projects/GenAIRAG
OLLAMA_HOST=http://127.0.0.1:11434 .venv/bin/python rag_app.py ingest --folder ./data
```

## 6) Ask questions
```bash
cd /Users/sureshcheppalli/Projects/GenAIRAG
OLLAMA_HOST=http://127.0.0.1:11434 .venv/bin/python rag_app.py ask --question "What are the key points?"
```

## Optional model override
```bash
CHAT_MODEL=llama3.2:3b OLLAMA_HOST=http://127.0.0.1:11434 .venv/bin/python rag_app.py ask --question "..."
```
