# storage/abstract_store.py
import os
import sqlite3
import faiss
import numpy as np
from config import ABSTRACT_DB_PATH, ABSTRACT_FAISS_PATH, EMBEDDING_DIM

class AbstractVectorStore:
    def __init__(self):
        os.makedirs(os.path.dirname(ABSTRACT_DB_PATH), exist_ok=True)

        # SQLite
        self.conn = sqlite3.connect(ABSTRACT_DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()

        # FAISS
        self.index = faiss.IndexIDMap(
            faiss.IndexFlatL2(EMBEDDING_DIM)
        )
        if os.path.exists(ABSTRACT_FAISS_PATH):
            self.index = faiss.read_index(ABSTRACT_FAISS_PATH)

    def _init_db(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS abstracts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arxiv_id TEXT,
                title TEXT,
                abstract TEXT,
                categories TEXT,
                update_date TEXT
            )
        """)
        self.conn.commit()

    def add_documents(self, records, embeddings):
        ids = []
        for r in records:
            self.cursor.execute("""
                INSERT INTO abstracts (arxiv_id, title, abstract, categories, update_date)
                VALUES (?, ?, ?, ?, ?)
            """, (
                r["arxiv_id"],
                r["title"],
                r["abstract"],
                r["categories"],
                r["update_date"]
            ))
            ids.append(self.cursor.lastrowid)

        self.conn.commit()

        vectors = np.array(embeddings).astype("float32")
        ids_np = np.array(ids).astype("int64")
        self.index.add_with_ids(vectors, ids_np)

        faiss.write_index(self.index, ABSTRACT_FAISS_PATH)

    def search(self, query_vector, k=5):
        q = np.array([query_vector]).astype("float32")
        distances, ids = self.index.search(q, k)

        valid_ids = [int(i) for i in ids[0] if i != -1]
        if not valid_ids:
            return []

        placeholders = ",".join("?" * len(valid_ids))
        sql = f"""
            SELECT arxiv_id, title, abstract
            FROM abstracts
            WHERE id IN ({placeholders})
        """
        self.cursor.execute(sql, valid_ids)
        return self.cursor.fetchall()
