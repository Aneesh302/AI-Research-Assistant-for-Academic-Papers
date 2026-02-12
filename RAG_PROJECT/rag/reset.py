# rag/reset.py
from storage.pdf_store import PDFVectorStore

def reset_pdf_context(store: PDFVectorStore):
    store.clear()
