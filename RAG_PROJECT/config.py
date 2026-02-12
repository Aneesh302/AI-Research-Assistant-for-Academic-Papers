# config.py
import os

# ========== AWS SETTINGS ==========
# We don't need to put keys here because 'aws configure' handles it!
BEDROCK_REGION = "us-east-1" 
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0" 

# ========== EMBEDDING ==========
EMBEDDING_DIM = 1024  # Titan V2 is 1024 dimensions
BATCH_SIZE = 50       # Bedrock is fast, we can do larger batches

# ========== GOOGLE API ==========
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

LLM_MODEL = "gemini-2.5-flash"

# ========== STORAGE ==========
DATA_DIR = "./data"

ABSTRACT_DB_PATH = os.path.join(DATA_DIR, "abstracts.db")
PDF_DB_PATH = os.path.join(DATA_DIR, "pdf_chunks.db")

ABSTRACT_FAISS_PATH = os.path.join(DATA_DIR, "abstract_index.faiss")
PDF_FAISS_PATH = os.path.join(DATA_DIR, "pdf_index.faiss")


# ========== PDF ==========
PDF_TMP_PATH = os.path.join(DATA_DIR, "temp.pdf")


