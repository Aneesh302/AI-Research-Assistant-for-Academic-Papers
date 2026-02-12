# rag/search.py
from storage.abstract_store import AbstractVectorStore
from embeddings.embedder import BedrockEmbedder

class AbstractSearchService:
    # UPDATED: Type hint changed to BedrockEmbedder
    def __init__(self, embedder: BedrockEmbedder, store: AbstractVectorStore):
        self.embedder = embedder
        self.store = store

    def search(self, query: str, k: int = 5):
        """
        Returns top-k papers (arxiv_id, title, abstract)
        """
        # Embed query using AWS Bedrock
        query_vec = self.embedder.embed_query(query)
        
        # Perform similarity search
        results = self.store.search(query_vec, k=k)

        formatted = []
        for arxiv_id, title, abstract in results:
            formatted.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract
            })

        return formatted