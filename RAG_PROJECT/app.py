# app.py
import streamlit as st
import os

# ------------------------------------------------
# 1. PAGE CONFIG (MUST BE FIRST)
# ------------------------------------------------
st.set_page_config(
    page_title="Academic Research Assistant", 
    layout="wide"
)

from google import genai
from config import GOOGLE_API_KEY

# UPDATED: Use Bedrock for Embeddings
from embeddings.embedder import BedrockEmbedder

from storage.abstract_store import AbstractVectorStore
from storage.pdf_store import PDFVectorStore

from rag.search import AbstractSearchService
from rag.pdf_ingest import PDFIngestService
from rag.qa import PDFQuestionAnsweringService
from rag.reset import reset_pdf_context

# ML Classification
from ml_classifier import PaperClassifier

# ------------------------------------------------
# ENVIRONMENT SETUP
# ------------------------------------------------
# We still need Google API Key for the LLM (Gemini 2.0 Flash)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def create_genai_client():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    return genai.Client(api_key=GOOGLE_API_KEY)

# Initialize Google Client for Text Generation only
genai_client = create_genai_client()

# ------------------------------------------------
# INITIALIZE SERVICES (ONCE)
# ------------------------------------------------
@st.cache_resource
def init_services():
    # 1. EMBEDDER: AWS Bedrock (No API key needed here, uses local AWS config)
    embedder = BedrockEmbedder()
    
    # 2. STORES: Initialize Database connections
    abstract_store = AbstractVectorStore()
    pdf_store = PDFVectorStore()

    # 3. SERVICES: Inject dependencies
    # Search uses Bedrock Embeddings + Abstract Store
    search_service = AbstractSearchService(embedder, abstract_store)
    
    # Ingest uses Bedrock Embeddings + PDF Store
    pdf_ingest_service = PDFIngestService(embedder, pdf_store)
    
    # QA uses Bedrock Embeddings (for retrieval) + Google Gemini (for answer generation)
    qa_service = PDFQuestionAnsweringService(embedder, pdf_store, genai_client)
    
    # 4. ML CLASSIFIER: Initialize paper category classifier
    classifier = PaperClassifier()

    return search_service, pdf_ingest_service, qa_service, pdf_store, classifier

search_service, pdf_ingest_service, qa_service, pdf_store, classifier = init_services()

# ------------------------------------------------
# STREAMLIT SESSION STATE
# ------------------------------------------------
if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "selected_paper" not in st.session_state:
    st.session_state.selected_paper = None

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# ------------------------------------------------
# UI HEADER
# ------------------------------------------------

st.title("Academic Paper Research Assistant")
st.markdown(
    """
This application helps you:
-  Discover academic papers using semantic search (**AWS Bedrock**)
-  Load and analyze full PDFs  
-  Ask deep questions using Retrieval-Augmented Generation (**Gemini 2.0**)
-  Classify paper abstracts into categories using **Machine Learning**

Built with **FAISS + SQLite + AWS Bedrock + Gemini + Scikit-learn**.
"""
)

# ------------------------------------------------
# SECTION 1 — ABSTRACT SEARCH
# ------------------------------------------------
st.header("🔍 Search Research Papers")

query = st.text_input(
    "Enter a research topic (e.g., videos, diffusion, transformers)",
    placeholder="e.g. video generation, diffusion models"
)

if st.button("Search Papers"):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching abstracts..."):
            results = search_service.search(query, k=5)

        if not results:
            st.info("No papers found.")
        else:
            st.session_state.search_results = results
            # Reset selection when new search is performed
            st.session_state.selected_paper = None
            st.session_state.pdf_loaded = False

# ------------------------------------------------
# SECTION 2 — DISPLAY SEARCH RESULTS
# ------------------------------------------------
if st.session_state.search_results:
    st.subheader("Search Results")

    titles = [
        f"{i+1}. {r['title']}"
        for i, r in enumerate(st.session_state.search_results)
    ]

    selected_index = st.radio(
        "Select a paper to explore:",
        options=range(len(titles)),
        format_func=lambda x: titles[x]
    )

    if st.button("Load Selected Paper"):
        selected = st.session_state.search_results[selected_index]
        arxiv_id = selected["arxiv_id"]

        with st.spinner("Downloading and embedding PDF (cached if available)..."):
            # This will now use Bedrock Embeddings to process the PDF
            pdf_ingest_service.ingest(arxiv_id)

        st.session_state.selected_paper = selected
        st.session_state.pdf_loaded = True

        st.success(f"PDF loaded for paper: **{selected['title']}**")

# ------------------------------------------------
# SECTION 3 — PDF QUESTION ANSWERING
# ------------------------------------------------
if st.session_state.pdf_loaded:
    st.header("Ask Questions About the Paper")

    st.markdown(
        f"**Selected Paper:** {st.session_state.selected_paper['title']}"
    )

    question = st.text_input(
        "Enter your question about this paper:",
        placeholder="How does the proposed method work?"
    )

    if st.button("Answer Question"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer using RAG..."):
                # 1. Retrieval: Uses Bedrock Embeddings
                # 2. Generation: Uses Gemini 2.0 Flash
                answer = qa_service.answer(question)

            st.markdown("### Answer")
            st.write(answer)

# ------------------------------------------------
# SECTION 4 — PAPER CATEGORY CLASSIFICATION
# ------------------------------------------------
st.divider()
st.header("🤖 Classify Paper Category")

st.markdown(
    """
    Enter an abstract (with optional title) to classify it into one of 8 academic categories:
    **bio, cs, econ, engineering, math, other, physics, stat**
    
    Uses a trained Logistic Regression model with TF-IDF features.
    """
)

abstract_text = st.text_area(
    "Enter paper title and/or abstract:",
    placeholder="Example: Deep Learning for Computer Vision. We present a novel convolutional neural network architecture...",
    height=150
)

if st.button("Classify Paper"):
    if not abstract_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Classifying paper category..."):
            try:
                # Get detailed classification results
                results = classifier.classify_with_details(abstract_text, top_k=3)
                
                # Display main prediction
                st.success(f"**Predicted Category:** {results['predicted_category'].upper()}")
                st.metric(
                    label="Confidence",
                    value=f"{results['confidence']*100:.1f}%"
                )
                
                # Display top 3 predictions
                st.markdown("#### Top 3 Predictions")
                
                for i, pred in enumerate(results['top_predictions'], 1):
                    category = pred['category']
                    probability = pred['probability']
                    
                    # Create a progress bar for each prediction
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.progress(probability, text=f"{i}. **{category.upper()}**")
                    with col2:
                        st.write(f"{probability*100:.1f}%")
                
            except Exception as e:
                st.error(f"Classification failed: {str(e)}")
                st.info("Make sure the ML model files are available in the correct location.")

# ------------------------------------------------
# SECTION 5 — RESET CONTEXT
# ------------------------------------------------
st.divider()

if st.button("🔄 Start New Topic"):
    reset_pdf_context(pdf_store)
    st.session_state.search_results = []
    st.session_state.selected_paper = None
    st.session_state.pdf_loaded = False
    st.success("Context cleared. You can start a new search.")