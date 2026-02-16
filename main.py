import os
import warnings
import streamlit as st
import streamlit.components.v1 as components
import tempfile
import re
import base64
from typing import Dict

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


try:
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*deprecated.*', module='tensorflow')
except Exception:
    pass


import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch.classes.*')

from pdf_processor import PDFProcessor
from rag_engine import RAGEngine
try:
    from highlighter import PDFHighlighter  
except Exception:
    PDFHighlighter = None  
from simple_highlighter import SimpleHighlighter
from perplexity_highlighter import PerplexityHighlighter

# NEW imports for numerical solving mode
from numerical_solver import NumericalSolver
from numerical_rag_engine import NumericalRAGEngine
from numerical_interface import render_numerical_mode
from enhanced_numerical_interface import render_enhanced_numerical_mode

# Existing imports for question-solving mode
try:
    from question_extractor import QuestionExtractor
    from question_solver import QuestionSolver
    from report_generator import ReportGenerator
    _HAS_QUESTION_SOLVER = True
except ImportError:
    QuestionExtractor = None
    QuestionSolver = None 
    ReportGenerator = None
    _HAS_QUESTION_SOLVER = False

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:
    get_script_run_ctx = None

if load_dotenv is not None:
    load_dotenv()

if get_script_run_ctx is not None and get_script_run_ctx() is None:
    print("This is a Streamlit app. Run it with: streamlit run main.py")
    raise SystemExit(1)

st.set_page_config(page_title="RAG PDF Highlighter", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
      html, body, [class*="css"], [class*="st"] { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
      .block-container { padding-top: 1.4rem; padding-bottom: 1.6rem; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      .stTextInput > div > div > input { border-radius: 10px; }
      .stButton > button { border-radius: 10px; padding: 0.55rem 0.9rem; font-weight: 600; }
      .stExpander { border-radius: 12px; overflow: hidden; }
      .minimal-card { border: 1px solid rgba(49, 51, 63, 0.12); border-radius: 14px; padding: 14px 14px; background: rgba(255,255,255,0.7); }
      .pdf-pane { border: 1px solid rgba(49, 51, 63, 0.12); border-radius: 14px; background: rgba(255,255,255,0.7); overflow: hidden; }
      .pdf-pane-inner { transition: max-height 320ms ease, opacity 260ms ease; max-height: 0px; opacity: 0; }
      .pdf-pane-inner.open { max-height: 1200px; opacity: 1; }
      .pdf-pane-header { padding: 12px 14px; border-bottom: 1px solid rgba(49, 51, 63, 0.10); display: flex; justify-content: space-between; align-items: center; }
      .pdf-pane-body { padding: 0; }
      .pdf-iframe { width: 100%; height: 78vh; border: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = {}
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'numerical_rag_engine' not in st.session_state:
    st.session_state.numerical_rag_engine = None
if 'numerical_solver' not in st.session_state:
    st.session_state.numerical_solver = None
if 'index_built_for_total_chunks' not in st.session_state:
    st.session_state.index_built_for_total_chunks = 0
# Question-solving state
if 'question_doc_path' not in st.session_state:
    st.session_state.question_doc_path = None
if 'extracted_questions' not in st.session_state:
    st.session_state.extracted_questions = []
if 'question_solutions' not in st.session_state:
    st.session_state.question_solutions = []
if 'preview_pdf_b64' not in st.session_state:
    st.session_state.preview_pdf_b64 = None
if 'preview_pdf_page' not in st.session_state:
    st.session_state.preview_pdf_page = 1
if 'preview_pdf_title' not in st.session_state:
    st.session_state.preview_pdf_title = None

st.title("🔍 RAG PDF Highlighter System")
st.sidebar.title("PDF Management")

perplexity_api_key = os.getenv("PERPLEXITY_API_KEY", "")
if perplexity_api_key:
    st.sidebar.success("Perplexity API key loaded from environment")
else:
    st.sidebar.info("Perplexity API key not set (using local/fallback mode)")


highlight_engine = st.sidebar.radio(
    "Highlighting Engine",
    ["Keyword-based (local)", "Perplexity (LLM)"],
    index=0,
)
if highlight_engine == "Perplexity (LLM)" and not perplexity_api_key:
    st.sidebar.warning("Set PERPLEXITY_API_KEY in your environment/.env to use LLM-based highlighting.")

# Updated mode selector with new Numerical Mode
mode_options = ["Standard Q&A", "Numerical Mode", "Enhanced Numerical"]
if _HAS_QUESTION_SOLVER:
    mode_options.append("Question-Solving Mode")

mode = st.sidebar.radio(
    "Mode",
    mode_options,
    index=0,
    help="Standard Q&A: Text-based questions | Numerical Mode: Math problems using formulas | Question-Solving Mode: Batch processing"
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    processor = PDFProcessor()
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_pdfs:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            with st.spinner(f"Processing {uploaded_file.name}..."):
                chunks = processor.extract_and_chunk(tmp_path, uploaded_file.name)
                st.session_state.processed_pdfs[uploaded_file.name] = {
                    'path': tmp_path,
                    'chunks': chunks
                }
            st.sidebar.success(f"✅ {uploaded_file.name} processed")


    all_chunks = []
    for pdf_data in st.session_state.processed_pdfs.values():
        all_chunks.extend(pdf_data['chunks'])

    # Build text-based RAG index for Standard Q&A mode
    if all_chunks and (not st.session_state.rag_engine or st.session_state.index_built_for_total_chunks != len(all_chunks)):
        with st.spinner("Building text search index..."):
            st.session_state.rag_engine = RAGEngine()
            st.session_state.rag_engine.build_index(all_chunks)
            st.session_state.index_built_for_total_chunks = len(all_chunks)

    # Initialize numerical solver for Numerical Mode
    if mode in ("Numerical Mode", "Enhanced Numerical"):
        if st.session_state.numerical_solver is None:
            with st.spinner("Initializing numerical solver..."):
                st.session_state.numerical_solver = NumericalSolver()
                st.session_state.numerical_rag_engine = NumericalRAGEngine()
                st.sidebar.success("✅ Numerical solver ready")


# Standard Q&A Mode (existing text-based system)
if mode == "Standard Q&A":
    if st.session_state.processed_pdfs:
        col_left, col_right = st.columns([0.58, 0.42], gap="large")

        with col_left:
            st.markdown('<div class="minimal-card">', unsafe_allow_html=True)
            st.header("Ask Questions")
            query = st.text_input("Enter your question:")
            show_debug = st.checkbox("Show retrieval details (scores & quotes)", value=False)
            restrict_to_answer = st.checkbox("Make highlighting follow this answer exactly (no extra context)", value=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if query and st.session_state.rag_engine:
                with st.spinner("Generating answer..."):

                    if show_debug:
                        try:
                            encoder_name = type(st.session_state.rag_engine.encoder).__name__
                            total_chunks = len(getattr(st.session_state.rag_engine, "chunks", []) or [])
                            st.caption(f"Debug: encoder={encoder_name} | total_chunks={total_chunks}")
                            dbg = st.session_state.rag_engine.debug_search(query, top_n=5)
                            with st.expander("Retrieval debug (top matches)", expanded=True):
                                st.json(dbg)
                        except Exception as e:
                            st.caption(f"Debug unavailable: {e}")

                    response = st.session_state.rag_engine.query(query)

                    st.subheader("Answer")
                    st.write(response['answer'])

                    st.subheader("Sources")
                    for i, source in enumerate(response['sources'][:3]):
                        with st.expander(f"Source {i+1}: {source['filename']} (Page {source['page']})"):
                            if show_debug:
                                st.caption(f"Relevance: {source['relevance_score']:.3f}")
                                st.code(source.get('quote', '') or source['text'][:200], language=None)
                            st.write(source['text'])

                    def _select_answer_based_snippet(source_text: str, answer_text: str) -> str:
                        try:
                            ans_words = [w for w in re.split(r"[^A-Za-z0-9]+", (answer_text or "").lower()) if len(w) > 3]
                            ans_set = set(ans_words)
                            if not source_text:
                                return ""
                            sentences = re.split(r"(?<=[.!?])\s+", source_text)
                            best_sentence = ""
                            best_score = -1
                            for s in sentences:
                                s_l = s.lower()
                                s_words = [w for w in re.split(r"[^A-Za-z0-9]+", s_l) if len(w) > 3]
                                if not s_words:
                                    continue
                                overlap = len(set(s_words) & ans_set)

                                score = overlap * 10 - max(0, len(s) - 220) / 80.0
                                if score > best_score:
                                    best_score = score
                                    best_sentence = s.strip()
                            if best_sentence:
                                return best_sentence[:300]
                            return source_text[:240]
                        except Exception:
                            return (source_text or "")[:240]

                    def _build_targets_for_preview(resp: Dict) -> tuple[str | None, int | None, list]:
                        try:
                            ans = resp.get('answer', '') or ''
                            if not resp.get('sources'):
                                return None, None, []
                            top = resp['sources'][0]
                            filename = top.get('filename')
                            page_one_idx = int(top.get('page', 1))
                            snippet = top.get('quote') or _select_answer_based_snippet(top.get('text', ''), ans)
                            if not filename or not snippet:
                                return None, None, []
                            return filename, page_one_idx, [(page_one_idx - 1, snippet)]
                        except Exception:
                            return None, None, []

                    def _generate_preview_pdf(resp: Dict):
                        filename, page_one_idx, targets = _build_targets_for_preview(resp)
                        if not filename or not targets:
                            st.session_state.preview_pdf_b64 = None
                            st.session_state.preview_pdf_title = None
                            return
                        pdf_path = st.session_state.processed_pdfs.get(filename, {}).get('path')
                        if not pdf_path:
                            st.session_state.preview_pdf_b64 = None
                            st.session_state.preview_pdf_title = None
                            return
                        out_stub = os.path.splitext(filename)[0]
                        highlighted_path = None
                        try:
                            if highlight_engine == "Perplexity (LLM)":
                                highlighted_path = PerplexityHighlighter().highlight_multiple_perplexity(
                                    pdf_path,
                                    targets,
                                    resp.get('answer', ''),
                                    out_stub=out_stub,
                                )
                            else:
                                highlighted_path = SimpleHighlighter().highlight_multiple_simple(
                                    pdf_path,
                                    targets,
                                    resp.get('answer', ''),
                                    out_stub=out_stub,
                                )
                        except Exception:
                            highlighted_path = None
                        if highlighted_path and os.path.exists(highlighted_path):
                            try:
                                with open(highlighted_path, 'rb') as f:
                                    b64 = base64.b64encode(f.read()).decode('utf-8')
                                st.session_state.preview_pdf_b64 = b64
                                st.session_state.preview_pdf_page = int(page_one_idx or 1)
                                st.session_state.preview_pdf_title = f"{filename} (Page {int(page_one_idx or 1)})"
                            except Exception:
                                st.session_state.preview_pdf_b64 = None
                                st.session_state.preview_pdf_title = None
                        else:
                            st.session_state.preview_pdf_b64 = None
                            st.session_state.preview_pdf_title = None

                    _generate_preview_pdf(response)

        with col_right:
            has_preview = bool(st.session_state.preview_pdf_b64)
            pane_title = st.session_state.preview_pdf_title or "Highlighted Preview"
            inner_cls = "pdf-pane-inner open" if has_preview else "pdf-pane-inner"
            components.html(
                f"""
                <div class="pdf-pane">
                  <div class="pdf-pane-header">
                    <div style="font-weight:600;">{pane_title}</div>
                    <div style="opacity:0.7; font-size:12px;">{('Ready' if has_preview else 'Ask a question to preview')}</div>
                  </div>
                  <div class="pdf-pane-body">
                    <div class="{inner_cls}">
                      {'' if not has_preview else f'<iframe class="pdf-iframe" src="data:application/pdf;base64,{st.session_state.preview_pdf_b64}#page={int(st.session_state.preview_pdf_page or 1)}"></iframe>'}
                    </div>
                  </div>
                </div>
                """,
                height=900,
                scrolling=True,
            )
    else:
        st.info("Upload one or more PDFs to get started.")

# Numerical Mode - Advanced numerical solver for probability and statistics
elif mode == "Numerical Mode":
    render_numerical_mode()
elif mode == "Enhanced Numerical":
    render_enhanced_numerical_mode()

# Question-Solving Mode (existing batch processing system)
elif mode == "Question-Solving Mode" and _HAS_QUESTION_SOLVER:
    st.header("🧠 Question-Solving Mode")
    if not st.session_state.processed_pdfs:
        st.info("Upload one or more source PDFs in the sidebar to build the knowledge base.")

    # Upload a question document
    qdoc = st.file_uploader("Upload a question document (PDF)", type=["pdf"], accept_multiple_files=False, key="qdoc")
    if qdoc is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_q:
            tmp_q.write(qdoc.read())
            st.session_state.question_doc_path = tmp_q.name
            st.session_state.extracted_questions = []
            st.session_state.question_solutions = []
            st.success(f"Loaded question document: {qdoc.name}")

    # Extract questions
    col_a, col_b = st.columns(2)
    with col_a:
        extract_btn = st.button("🔎 Extract Questions", disabled=not st.session_state.question_doc_path)
    with col_b:
        clear_btn = st.button("Clear Extracted")
        if clear_btn:
            st.session_state.extracted_questions = []
            st.session_state.question_solutions = []

    if extract_btn and st.session_state.question_doc_path:
        extractor = QuestionExtractor()
        with st.spinner("Extracting questions..."):
            res = extractor.extract_questions_from_pdf(st.session_state.question_doc_path)
            st.session_state.extracted_questions = res.get("questions", [])
            if not st.session_state.extracted_questions:
                st.warning("No questions detected. You can still type one manually below.")

    # Manual add
    with st.expander("Add a question manually"):
        manual_q = st.text_input("Question text", key="manual_q_text")
        manual_type = st.selectbox("Type", ["definition", "explanation", "list", "compare", "calculation", "fact"], index=1)
        if st.button("Add Question"):
            if manual_q.strip():
                new_id = (st.session_state.extracted_questions[-1]['id'] + 1) if st.session_state.extracted_questions else 1
                st.session_state.extracted_questions.append({
                    "id": new_id,
                    "text": manual_q.strip(),
                    "page": 0,
                    "type": manual_type,
                    "context": "",
                    "confidence": 1.0,
                })

    # Display extracted
    if st.session_state.extracted_questions:
        st.subheader("Detected Questions")
        selected_ids = []
        for q in st.session_state.extracted_questions:
            cols = st.columns([0.05, 0.65, 0.1, 0.1, 0.1])
            with cols[0]:
                chk = st.checkbox("", key=f"qsel_{q['id']}")
                if chk:
                    selected_ids.append(q['id'])
            with cols[1]:
                st.text_input("", value=q['text'], key=f"qt_{q['id']}")
            with cols[2]:
                st.number_input("Page", value=int(q.get('page', 0)), key=f"qp_{q['id']}", step=1)
            with cols[3]:
                st.selectbox("Type", ["definition", "explanation", "list", "compare", "calculation", "fact"], index=['definition','explanation','list','compare','calculation','fact'].index(q.get('type','explanation')), key=f"qy_{q['id']}")
            with cols[4]:
                st.number_input("Conf", value=float(q.get('confidence', 0.7)), key=f"qc_{q['id']}", step=0.05)

        # Sync edits back to objects
        for q in st.session_state.extracted_questions:
            q['text'] = st.session_state.get(f"qt_{q['id']}", q['text'])
            q['page'] = int(st.session_state.get(f"qp_{q['id']}", q.get('page', 0)))
            q['type'] = st.session_state.get(f"qy_{q['id']}", q.get('type','explanation'))
            q['confidence'] = float(st.session_state.get(f"qc_{q['id']}", q.get('confidence', 0.7)))

        solve_col1, solve_col2, solve_col3 = st.columns([0.25, 0.25, 0.5])
        with solve_col1:
            btn_solve_sel = st.button("✅ Solve Selected", disabled=(not selected_ids or not st.session_state.rag_engine))
        with solve_col2:
            btn_solve_all = st.button("Solve All", disabled=(not st.session_state.extracted_questions or not st.session_state.rag_engine))

        if (btn_solve_sel or btn_solve_all) and st.session_state.rag_engine:
            runner = QuestionSolver()
            to_solve = [q for q in st.session_state.extracted_questions if (q['id'] in selected_ids) or btn_solve_all]
            solutions = []
            with st.spinner("Solving questions..."):
                for q in to_solve:
                    sol = runner.solve(q, st.session_state.rag_engine)
                    solutions.append({"question": q, "solution": sol})
            st.session_state.question_solutions = solutions

    # Show solutions
    if st.session_state.question_solutions:
        st.subheader("Solutions")
        for i, item in enumerate(st.session_state.question_solutions, 1):
            q = item['question']
            s = item['solution']
            st.markdown(f"**Q{i}.** {q.get('text','')}")
            st.write(s.get('answer',''))
            meta = f"Confidence: {s.get('confidence',0.0)} | Out of scope: {s.get('out_of_scope')}"
            st.caption(meta)
            if s.get('sources'):
                with st.expander("Sources"):
                    for src in s['sources']:
                        st.write(f"- {src.get('filename')} (Page {src.get('page')}) — score {src.get('relevance_score',0.0):.3f}")

        # Export
        st.subheader("Export")
        rg = ReportGenerator()
        colx, coly = st.columns(2)
        with colx:
            if st.button("📄 Download PDF Report"):
                path = rg.save_pdf_report(st.session_state.question_solutions, basename="solutions")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button("Download solutions.pdf", f.read(), file_name=os.path.basename(path), mime="application/pdf")
        with coly:
            if st.button("🧾 Download JSON"):
                pathj = rg.save_json(st.session_state.question_solutions, basename="solutions")
                if os.path.exists(pathj):
                    with open(pathj, "rb") as f:
                        st.download_button("Download solutions.json", f.read(), file_name=os.path.basename(pathj), mime="application/json")

elif mode == "Question-Solving Mode" and not _HAS_QUESTION_SOLVER:
    st.error("Question-Solving Mode is not available. Missing required modules.")

if st.session_state.processed_pdfs:
    st.sidebar.subheader("Uploaded PDFs")
    for filename in st.session_state.processed_pdfs.keys():
        st.sidebar.text(f"📄 {filename}")
