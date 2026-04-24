import argparse
import hashlib
import os
from pathlib import Path

import chromadb
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Load .env FIRST so all os.getenv() calls below pick up the values
_MITWORK_ENV = Path(__file__).resolve().parents[1] / "MITWork" / ".env"
load_dotenv(_MITWORK_ENV if _MITWORK_ENV.exists() else None)

OPENAI_MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHROMA_DIR         = Path(os.getenv("CHROMA_DIR", "./chroma_db")).resolve()
COLLECTION_NAME    = os.getenv("COLLECTION", "agentic_pdf_docs_openai")
TOP_K_DEFAULT      = int(os.getenv("TOP_K", "10"))   # chunks retrieved per query; override in .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("ERROR: Missing OPENAI_API_KEY / OPENAI_KEY. Set it in your .env file.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------
class ContextItem(BaseModel):
    source: str
    chunk: str

    def as_text(self, max_chars: int = 500) -> str:
        return f"[{self.source}]\n{self.chunk[:max_chars]}"


class AskResult(BaseModel):
    answer: str
    contexts: list[ContextItem]

    def as_text(self) -> str:
        """Return the full result as a clean, human-readable string."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("ANSWER")
        lines.append("=" * 60)
        lines.append(self.answer)
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"SUPPORTING CONTEXT  ({len(self.contexts)} chunk(s))")
        lines.append("=" * 60)
        for idx, ctx in enumerate(self.contexts, 1):
            lines.append(f"\n[{idx}] Source: {ctx.source}")
            lines.append("-" * 40)
            lines.append(ctx.chunk[:500])
        lines.append("")
        return "\n".join(lines)

    def as_markdown(self) -> str:
        """Return a Markdown-formatted string for Gradio rendering."""
        lines: list[str] = []
        lines.append(self.answer)
        lines.append("")
        lines.append("---")
        lines.append(f"**📄 Supporting context — {len(self.contexts)} chunk(s)**")
        for idx, ctx in enumerate(self.contexts, 1):
            lines.append(f"\n**[{idx}] {ctx.source}**")
            lines.append(f"```\n{ctx.chunk[:400]}\n```")
        return "\n".join(lines)


class IngestResult(BaseModel):
    files_processed: int
    chunks_ingested: int
    collection: str

    def as_text(self) -> str:
        return (
            f"Ingest complete.\n"
            f"  Files processed : {self.files_processed}\n"
            f"  Chunks ingested : {self.chunks_ingested}\n"
            f"  Collection      : {self.collection}"
        )


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Split *text* into overlapping chunks of at most *chunk_size* characters."""
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings in a single batched API call."""
    if not texts:
        return []
    response = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    items = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in items]


def generate_answer(prompt: str) -> str:
    """Call the chat-completion endpoint and return the assistant's reply."""
    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise RAG assistant. "
                    "Answer questions using only the provided context. "
                    "If the context is insufficient, say so clearly and briefly."
                    "Be concise and to the point. Keep your answer short and sweet."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return (completion.choices[0].message.content or "").strip()


def fallback_from_contexts(question: str, context_items: list[ContextItem]) -> str:
    """Produce a grounded fallback answer when the LLM says it doesn't know."""
    if not context_items:
        return "No relevant context found for that question."
    snippets = [c.chunk.replace("\n", " ").strip()[:320] for c in context_items[:4]]
    joined = " … ".join(snippets)
    return (
        f"The model could not give a definitive answer for: '{question}'. "
        f"Best grounded summary from retrieved context: {joined}"
    )


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------
def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------
def ingest(folder: str) -> IngestResult:
    """Read all .txt files in *folder*, chunk, embed, and store in Chroma."""
    path = Path(folder)
    if not path.exists():
        raise ValueError(f"Folder not found: {path}")

    files = sorted(path.glob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found in: {path}")

    docs:  list[str]  = []
    ids:   list[str]  = []
    metas: list[dict] = []

    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(chunk_text(text)):
            doc_id = hashlib.sha1(f"{f.name}:{i}:{chunk}".encode()).hexdigest()
            docs.append(chunk)
            ids.append(doc_id)
            metas.append({"source": f.name, "chunk_index": i})

    embeddings = embed_texts(docs)
    col = get_collection()
    col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    return IngestResult(
        files_processed=len(files),
        chunks_ingested=len(docs),
        collection=COLLECTION_NAME,
    )


def ask_core(question: str, k: int = TOP_K_DEFAULT) -> AskResult:
    """Embed the question, retrieve top-k chunks, generate an answer."""
    qvec = embed_texts([question])[0]
    col = get_collection()

    out   = col.query(query_embeddings=[qvec], n_results=k)
    docs  = (out.get("documents") or [[]])[0]
    metas = (out.get("metadatas")  or [[]])[0]

    if not docs:
        return AskResult(
            answer="No documents found in the collection. Please ingest some files first.",
            contexts=[],
        )

    context_items: list[ContextItem] = [
        ContextItem(
            source=metas[i].get("source", "unknown") if i < len(metas) else "unknown",
            chunk=d,
        )
        for i, d in enumerate(docs)
    ]

    context_block = "\n\n".join(
        f"[Source: {c.source}]\n{c.chunk}" for c in context_items
    )
    prompt = (
        "Answer using ONLY the context below.\n"
        "If the context is partial, provide the best grounded summary and note uncertainty.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\nAnswer:"
    )

    answer = generate_answer(prompt)

    if answer.strip().lower().rstrip(".") in {"i do not know", "i don't know"}:
        answer = fallback_from_contexts(question, context_items)

    return AskResult(answer=answer, contexts=context_items)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def gradio_ingest(folder_path: str) -> str:
    """Gradio handler: ingest .txt files from the given folder path."""
    folder_path = folder_path.strip()
    if not folder_path:
        return "⚠️ Please provide a folder path."
    try:
        result = ingest(folder_path)
        return f"✅ {result.as_text()}"
    except ValueError as e:
        return f"❌ {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"


def gradio_ingest_files(files: list, progress=gr.Progress()) -> str:
    """Gradio handler: ingest uploaded .txt files directly via the UI."""
    if not files:
        return "⚠️ No files uploaded."

    docs:  list[str]  = []
    ids:   list[str]  = []
    metas: list[dict] = []

    progress(0, desc="Reading uploaded files…")
    for file in files:
        file_path = Path(file.name)
        if file_path.suffix.lower() != ".txt":
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(chunk_text(text)):
            doc_id = hashlib.sha1(f"{file_path.name}:{i}:{chunk}".encode()).hexdigest()
            docs.append(chunk)
            ids.append(doc_id)
            metas.append({"source": file_path.name, "chunk_index": i})

    if not docs:
        return "❌ No valid .txt files found in the upload."

    progress(0.5, desc="Embedding chunks with OpenAI…")
    embeddings = embed_texts(docs)

    progress(0.85, desc="Storing in ChromaDB…")
    col = get_collection()
    col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    progress(1.0, desc="Done!")
    file_names = {m["source"] for m in metas}
    return (
        f"✅ Ingest complete.\n"
        f"  Files processed : {len(file_names)}\n"
        f"  Chunks ingested : {len(docs)}\n"
        f"  Collection      : {COLLECTION_NAME}"
    )


def gradio_chat(
    user_message: str,
    history: list[dict],
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    """
    Gradio chatbot handler.
    Returns (updated_history, updated_history) to keep chatbot + state in sync.
    """
    if not user_message.strip():
        return history, history

    try:
        result = ask_core(user_message.strip(), k=top_k)
        reply = result.as_markdown()
    except Exception as e:
        reply = f"❌ Error: {e}"

    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return history, history


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(
        title="RAG Chatbot — OpenAI + ChromaDB",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            """
            # 🤖 RAG Chatbot
            **Powered by OpenAI embeddings · GPT-4o · ChromaDB**

            Upload your `.txt` documents (or point to a server folder), then chat with them below.
            """
        )

        # ── Ingest tab ──────────────────────────────────────────────────────
        with gr.Tab("📥 Ingest Documents"):
            gr.Markdown("### Option 1 — Upload files directly")
            upload_box = gr.File(
                label="Upload .txt files",
                file_types=[".txt"],
                file_count="multiple",
            )
            upload_btn    = gr.Button("Ingest Uploaded Files", variant="primary")
            upload_status = gr.Textbox(label="Status", lines=5, interactive=False)

            upload_btn.click(
                fn=gradio_ingest_files,
                inputs=[upload_box],
                outputs=[upload_status],
            )

            gr.Markdown("### Option 2 — Ingest from a local folder path (server-side)")
            folder_input  = gr.Textbox(
                label="Folder path",
                placeholder="e.g.  ./data   or   /home/user/documents",
            )
            folder_btn    = gr.Button("Ingest Folder", variant="secondary")
            folder_status = gr.Textbox(label="Status", lines=5, interactive=False)

            folder_btn.click(
                fn=gradio_ingest,
                inputs=[folder_input],
                outputs=[folder_status],
            )

        # ── Chat tab ─────────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(
                label="RAG Chatbot",
                type="messages",
                height=520,
                show_copy_button=True,
                avatar_images=(None, "https://api.dicebear.com/9.x/bottts/svg?seed=rag"),
                render_markdown=True,
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask a question about your documents…",
                    label="Your question",
                    scale=8,
                    autofocus=True,
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=TOP_K_DEFAULT, step=1,
                    label="Top-K chunks",
                    scale=2,
                )

            with gr.Row():
                send_btn  = gr.Button("Send", variant="primary", scale=8)
                clear_btn = gr.Button("Clear chat", variant="secondary", scale=2)

            chat_state = gr.State([])

            # Submit on button click
            send_btn.click(
                fn=gradio_chat,
                inputs=[msg_box, chat_state, top_k_slider],
                outputs=[chatbot, chat_state],
            ).then(lambda: "", outputs=[msg_box])

            # Submit on Enter key
            msg_box.submit(
                fn=gradio_chat,
                inputs=[msg_box, chat_state, top_k_slider],
                outputs=[chatbot, chat_state],
            ).then(lambda: "", outputs=[msg_box])

            clear_btn.click(lambda: ([], []), outputs=[chatbot, chat_state])

        # ── Info footer ───────────────────────────────────────────────────────
        gr.Markdown(
            f"""
            ---
            **Active configuration**

            | Setting | Value |
            |---|---|
            | LLM model | `{OPENAI_MODEL}` |
            | Embedding model | `{OPENAI_EMBED_MODEL}` |
            | ChromaDB path | `{CHROMA_DIR}` |
            | Collection | `{COLLECTION_NAME}` |
            | Top-K (default) | `{TOP_K_DEFAULT}` |
            """
        )

    return demo


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------
def cmd_ingest(args: argparse.Namespace) -> None:
    result = ingest(args.folder)
    print(result.as_text())


def cmd_ask(args: argparse.Namespace) -> None:
    result = ask_core(args.question, args.k)
    print(result.as_text())


def cmd_ui(args: argparse.Namespace) -> None:
    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG App — OpenAI embeddings + GPT-4o + ChromaDB  (CLI & Gradio UI)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Chunk & embed .txt files into ChromaDB")
    p_ingest.add_argument("--folder", default="./data", help="Folder containing .txt files")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question against the indexed docs")
    p_ask.add_argument("--question", required=True, help="Your question")
    p_ask.add_argument("--k", type=int, default=TOP_K_DEFAULT, help=f"Top-k chunks to retrieve (default: {TOP_K_DEFAULT} from TOP_K env var)")

    # ui
    p_ui = sub.add_parser("ui", help="Launch the Gradio chatbot interface")
    p_ui.add_argument("--host",  default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_ui.add_argument("--port",  type=int, default=7860, help="Bind port (default: 7860)")
    p_ui.add_argument("--share", action="store_true", help="Create a public Gradio share link")

    args = parser.parse_args()
    {"ingest": cmd_ingest, "ask": cmd_ask, "ui": cmd_ui}[args.cmd](args)


if __name__ == "__main__":
    main()