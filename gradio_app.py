import gradio as gr

import rag_app


def ask(question: str, k: int) -> str:
    result = rag_app.ask_core(question, k)
    return result.answer


with gr.Blocks(title="GenAIRAG") as demo:
    gr.Markdown("## GenAIRAG")
    q = gr.Textbox(label="Question", placeholder="Ask about uploaded documents...")
    k = gr.Slider(minimum=1, maximum=20, value=4, step=1, label="Top K")
    out = gr.Textbox(label="Answer")
    btn = gr.Button("Ask")
    btn.click(fn=ask, inputs=[q, k], outputs=out)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
