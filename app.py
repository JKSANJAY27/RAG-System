import gradio as gr
import time
import json
import logging
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

from src.rag_pipeline import RAGPipeline
from src.ingestor import Document, get_ingestor

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize pipeline lazily to avoid heavy loading immediately
# Global pipeline instance
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

def format_json_block(data):
    """Safely format dict to JSON string with indentation."""
    if not data:
        return "{}"
    try:
        return json.dumps(data, indent=2)
    except:
        return str(data)

def ask_rag(question: str):
    """
    Query the RAG system and format the response to populate Gradio components.
    """
    if not question.strip():
        return (
            "Please enter a question.", 
            "No sources.", 
            "{}", "{}", "{}", "{}", "{}"
        )
    
    try:
        pipeline = get_pipeline()
        
        # This will error if no documents are ingested
        if pipeline.chunk_count == 0:
            return (
                "⚠️ ERROR: The knowledge base is empty! Please go to the 'Ingest Documents' tab and add some documents first.",
                "None", "{}", "{}", "{}", "{}", "{}"
            )
            
        start_time = time.time()
        response = pipeline.query(question)
        
        # 1. Format Answer
        answer = response.answer
        if response.citation_enforced:
            answer = f"⚠️ **Citation Enforcement Triggered:**\nThe model's confidence was too low based on the retrieved documents. It declined to answer to prevent hallucination.\n\n{answer}"
            
        # 2. Format Sources
        sources_text = ""
        if response.sources:
            for i, source in enumerate(response.sources, 1):
                sources_text += f"- [{i}] {source}\n"
        else:
            sources_text = "No direct sources cited."
            
        # 3. Extract Trace Details
        ctx = response.trace_ctx
        
        bm25_data = "{}"
        vector_data = "{}"
        rrf_data = "{}"
        rerank_data = "{}"
        metrics_data = "{}"
        
        if ctx:
            # BM25 Trace
            bm25_span = ctx.get_span("bm25")
            if bm25_span:
                bm25_data = format_json_block({
                    "latency_ms": bm25_span.get("latency_ms"),
                    "candidates_found": bm25_span.get("output", {}).get("candidates"),
                    "top_scores": bm25_span.get("output", {}).get("top_scores")
                })
                
            # Vector Trace
            vector_span = ctx.get_span("vector")
            if vector_span:
                vector_data = format_json_block({
                    "latency_ms": vector_span.get("latency_ms"),
                    "candidates_found": vector_span.get("output", {}).get("candidates"),
                    "top_scores": vector_span.get("output", {}).get("top_scores")
                })
                
            # RRF Trace
            rrf_span = ctx.get_span("rrf_fusion")
            if rrf_span:
                rrf_data = format_json_block({
                    "latency_ms": rrf_span.get("latency_ms"),
                    "bm25_count_in": rrf_span.get("input", {}).get("bm25_count"),
                    "vector_count_in": rrf_span.get("input", {}).get("vector_count"),
                    "fused_count_out": rrf_span.get("output", {}).get("fused_count"),
                    "top_rrf_scores": rrf_span.get("output", {}).get("top_rrf_scores")
                })
                
            # Re-rank Trace
            rerank_span = ctx.get_span("rerank")
            if rerank_span:
                # Include Before vs After
                out_dict = rerank_span.get("output", {})
                in_dict = rerank_span.get("input", {})
                
                rerank_info = {
                    "latency_ms": rerank_span.get("latency_ms"),
                    "citation_fired": out_dict.get("citation_enforced", False),
                    "top_score": out_dict.get("top_score", 0.0),
                    "before_order": in_dict.get("before_rerank", [])[:5],
                    "after_order": out_dict.get("after_rerank", [])[:5]
                }
                rerank_data = format_json_block(rerank_info)
                
            # Overall Metrics
            metrics = {
                "retrieval_latency_ms": response.retrieval_latency_ms,
                "generation_latency_ms": response.generation_latency_ms,
                "total_latency_ms": response.total_latency_ms,
                "model": response.model,
                "prompt_version": response.prompt_version,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "trace_id": response.trace_id
            }
            metrics_data = format_json_block(metrics)
            
        return (
            answer,
            sources_text,
            bm25_data,
            vector_data,
            rrf_data,
            rerank_data,
            metrics_data
        )

    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return (error_msg, "Error", "{}", "{}", "{}", "{}", "{}")

def ingest_doc(file_path, doc_type):
    """Handle document ingestion from the UI."""
    if not file_path:
        return "Please upload a file or provide a URL first."
    
    try:
        pipeline = get_pipeline()
        
        # For 'web', file_path is actually the URL string
        # For 'pdf'/'markdown', file_path is the temporary path to the uploaded file from Gradio
        
        if doc_type != "web" and hasattr(file_path, 'name'):
            # Gradio passes a temporary file object if gr.File is used. 
            # We need the path string.
            src_path = file_path.name
        else:
            src_path = file_path
            
        chunks_added = pipeline.ingest(src_path, doc_type)
        return f"✅ Successfully ingested document!\n\nAdded {chunks_added} chunks to the knowledge base.\nTotal chunks in system: {pipeline.chunk_count}"
    except Exception as e:
        import traceback
        return f"❌ Ingestion Error: {str(e)}\n\n{traceback.format_exc()}"

# ─── Gradio UI Layout ──────────────────────────────────────────────────────────

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif']
)

with gr.Blocks(theme=theme, title="RAG Explorer") as demo:
    gr.Markdown(
        """
        # 🔍 Transparent RAG System Explorer
        
        This interface demonstrates a production-grade **Retrieval Augmented Generation (RAG)** system built with Python, ChromaDB, and local Ollama LLMs.
        
        **What makes this special?** It completely unboxes the "black box" of RAG. Below your answer, you can explicitly see the outputs of every internal step: 
        Lexical BM25 Search → Semantic Vector Search → Reciprocal Rank Fusion → Cross-Encoder Re-Ranking → Citation Enforcement.
        """
    )
    
    with gr.Tabs():
        # Tab 1: Query System
        with gr.Tab("💬 Ask Question"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="Your Question", 
                        placeholder="e.g. What is the self-attention mechanism?",
                        lines=2
                    )
                    submit_btn = gr.Button("Submit Query", variant="primary")
                    
                    gr.Markdown("### 🤖 Answer")
                    answer_output = gr.Markdown(label="Generated Answer")
                    
                    gr.Markdown("### 📚 Cited Sources")
                    sources_output = gr.Markdown()
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 🔬 Pipeline Internals & Metrics")
                    gr.Markdown("*Expand these sections to see exactly how the RAG pipeline fetched internal context.*")
                    
                    with gr.Accordion("1. BM25 Keywords Search", open=False):
                        bm25_output = gr.Code(language="json", label="BM25 Lexical Trace")
                    
                    with gr.Accordion("2. Vector Semantic Search", open=False):
                        vector_output = gr.Code(language="json", label="Vector Search Trace")
                        
                    with gr.Accordion("3. Reciprocal Rank Fusion (RRF)", open=False):
                        rrf_output = gr.Code(language="json", label="RRF Merge Trace")
                        
                    with gr.Accordion("4. Cross-Encoder Re-Ranking", open=False):
                        gr.Markdown("Notice how chunks move UP or DOWN in the `after_order` compared to the `before_order`. This proves the Cross-Encoder is dynamically adjusting relevance.")
                        rerank_output = gr.Code(language="json", label="Before vs After Re-Ranking")
                        
                    with gr.Accordion("⏱️ Latency & Token Metrics", open=True):
                        metrics_output = gr.Code(language="json", label="Performance Overview")

        # Tab 2: Ingest Data
        with gr.Tab("📥 Ingest Documents"):
            gr.Markdown("Add new knowledge to the vector store. Supported types: PDF, Markdown, and Web URLs.")
            
            with gr.Row():
                with gr.Column():
                    doc_type_radio = gr.Radio(
                        ["markdown", "pdf", "web"], 
                        label="Document Type", 
                        value="markdown"
                    )
                    
                    # File upload for PDF/MD
                    file_upload = gr.File(label="Upload File (PDF or Markdown)")
                    
                    # Text box for Web URLs
                    url_input = gr.Textbox(label="Web URL", placeholder="https://en.wikipedia.org/wiki/...", visible=False)
                    
                    ingest_btn = gr.Button("Ingest into Knowledge Base", variant="secondary")
                    
                with gr.Column():
                    ingest_status = gr.Textbox(label="Ingestion Status", lines=10)

            # Toggle visibility of inputs based on document type
            def toggle_inputs(choice):
                if choice == "web":
                    return gr.update(visible=False), gr.update(visible=True)
                else:
                    return gr.update(visible=True), gr.update(visible=False)
                    
            doc_type_radio.change(
                fn=toggle_inputs,
                inputs=[doc_type_radio],
                outputs=[file_upload, url_input]
            )
            
            # Handle Ingestion based on type
            def process_ingest(d_type, f_up, u_in):
                target = u_in if d_type == "web" else f_up
                return ingest_doc(target, d_type)
                
            ingest_btn.click(
                fn=process_ingest,
                inputs=[doc_type_radio, file_upload, url_input],
                outputs=[ingest_status]
            )

    # Wire up the Submit button
    submit_btn.click(
        fn=ask_rag,
        inputs=[query_input],
        outputs=[
            answer_output,
            sources_output,
            bm25_output,
            vector_output,
            rrf_output,
            rerank_output,
            metrics_output
        ]
    )
    
    # Also support "Enter" key submission
    query_input.submit(
        fn=ask_rag,
        inputs=[query_input],
        outputs=[
            answer_output,
            sources_output,
            bm25_output,
            vector_output,
            rrf_output,
            rerank_output,
            metrics_output
        ]
    )

if __name__ == "__main__":
    import os
    # Pre-initialize the pipeline in the main thread to avoid PyTorch deadlocks on Windows
    # when loading models for the first time inside a Gradio background worker thread!
    print("⏳ Loading Models... Please wait!")
    get_pipeline()
    print("✅ Models loaded successfully!")
    
    # Expose the app
    # host="0.0.0.0" is needed for Docker to expose the port outside the container
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=False)
