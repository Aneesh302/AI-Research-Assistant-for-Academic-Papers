# storage/pdf_store.py
import os
import sqlite3
import faiss
import numpy as np
from config import PDF_DB_PATH, PDF_FAISS_PATH, EMBEDDING_DIM

class PDFVectorStore:
    def __init__(self):
        os.makedirs(os.path.dirname(PDF_DB_PATH), exist_ok=True)

        self.conn = sqlite3.connect(PDF_DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()

        self.index = faiss.IndexIDMap(
            faiss.IndexFlatL2(EMBEDDING_DIM)
        )
        if os.path.exists(PDF_FAISS_PATH):
            self.index = faiss.read_index(PDF_FAISS_PATH)

    def _init_db(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdf_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arxiv_id TEXT,
                page_number INTEGER,
                content TEXT
            )
        """)
        self.conn.commit()

    def has_pdf(self, arxiv_id):
        self.cursor.execute(
            "SELECT COUNT(*) FROM pdf_chunks WHERE arxiv_id = ?",
            (arxiv_id,)
        )
        return self.cursor.fetchone()[0] > 0

    def add_chunks(self, arxiv_id, chunks, embeddings):
        ids = []
        for i, text in enumerate(chunks):
            self.cursor.execute("""
                INSERT INTO pdf_chunks (arxiv_id, page_number, content)
                VALUES (?, ?, ?)
            """, (arxiv_id, i, text))
            ids.append(self.cursor.lastrowid)

        self.conn.commit()

        vectors = np.array(embeddings).astype("float32")
        ids_np = np.array(ids).astype("int64")
        self.index.add_with_ids(vectors, ids_np)

        faiss.write_index(self.index, PDF_FAISS_PATH)

    def search(self, query_vector, k=5):
        q = np.array([query_vector]).astype("float32")
        distances, ids = self.index.search(q, k)

        valid_ids = [int(i) for i in ids[0] if i != -1]
        if not valid_ids:
            return []

        placeholders = ",".join("?" * len(valid_ids))
        sql = f"""
            SELECT content FROM pdf_chunks
            WHERE id IN ({placeholders})
        """
        self.cursor.execute(sql, valid_ids)
        return [r[0] for r in self.cursor.fetchall()]

    def clear(self):
        self.cursor.execute("DELETE FROM pdf_chunks")
        self.conn.commit()
        self.index.reset()
        faiss.write_index(self.index, PDF_FAISS_PATH)
