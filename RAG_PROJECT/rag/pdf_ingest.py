# rag/pdf_ingest.py
import requests
import os
from langchain_community.document_loaders import PyPDFLoader

# UPDATED: Import BedrockEmbedder instead of GeminiEmbedder
from embeddings.embedder import BedrockEmbedder
from storage.pdf_store import PDFVectorStore
from config import PDF_TMP_PATH

class PDFIngestService:
    # UPDATED: Type hint changed to BedrockEmbedder
    def __init__(self, embedder: BedrockEmbedder, store: PDFVectorStore):
        self.embedder = embedder
        self.store = store

    def ingest(self, arxiv_id: str):
        """
        Downloads and embeds PDF if not already cached.
        """
        if self.store.has_pdf(arxiv_id):
            print(f"PDF {arxiv_id} already cached. Skipping embedding.")
            return

        print(f"Downloading PDF {arxiv_id}...")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        try:
            response = requests.get(pdf_url, timeout=30)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to download PDF (Status: {response.status_code})")

            with open(PDF_TMP_PATH, "wb") as f:
                f.write(response.content)

            # Load PDF (PyPDFLoader splits by page by default)
            loader = PyPDFLoader(PDF_TMP_PATH)
            docs = loader.load()

            # Extract text content
            texts = [doc.page_content for doc in docs]
            
            # Sanity check: Remove empty pages to save cost
            texts = [t for t in texts if len(t.strip()) > 50]

            if not texts:
                print(f"Warning: PDF {arxiv_id} seems empty or unreadable.")
                return

            print(f"Embedding {len(texts)} pages for PDF {arxiv_id}...")
            
            # Bedrock Embedding
            embeddings = self.embedder.embed_documents(texts)

            # Store in DB
            self.store.add_chunks(arxiv_id, texts, embeddings)
            print(f"Successfully ingested PDF {arxiv_id}")

        except Exception as e:
            print(f"Error ingesting PDF {arxiv_id}: {e}")
        finally:
            # Cleanup temp file
            if os.path.exists(PDF_TMP_PATH):
                os.remove(PDF_TMP_PATH)