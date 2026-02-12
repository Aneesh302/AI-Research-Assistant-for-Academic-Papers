# rag/pdf_ingest.py
import requests
from langchain_community.document_loaders import PyPDFLoader
from embeddings.embedder import GeminiEmbedder
from storage.pdf_store import PDFVectorStore
from config import PDF_TMP_PATH

class PDFIngestService:
    def __init__(self, embedder: GeminiEmbedder, store: PDFVectorStore):
        self.embedder = embedder
        self.store = store

    def ingest(self, arxiv_id: str):
        """
        Downloads and embeds PDF if not already cached.
        """
        if self.store.has_pdf(arxiv_id):
            print(f"PDF {arxiv_id} already cached. Skipping embedding.")
            return

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(pdf_url, timeout=30)

        if response.status_code != 200:
            raise RuntimeError("Failed to download PDF")

        with open(PDF_TMP_PATH, "wb") as f:
            f.write(response.content)

        loader = PyPDFLoader(PDF_TMP_PATH)
        docs = loader.load()

        texts = [doc.page_content for doc in docs]

        print(f"Embedding {len(texts)} PDF chunks...")
        embeddings = self.embedder.embed_documents(texts)

        self.store.add_chunks(arxiv_id, texts, embeddings)
