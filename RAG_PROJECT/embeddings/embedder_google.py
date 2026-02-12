# embeddings/embedder.py
import time
from typing import List
from google.genai import types
from google.api_core import retry
from config import EMBEDDING_MODEL, BATCH_SIZE

is_retriable = lambda e: hasattr(e, "code") and e.code in {429, 503}

class GeminiEmbedder:
    def __init__(self, genai_client):
        self.client = genai_client

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="retrieval_document"
            )
        )
        return [emb.values for emb in response.embeddings]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        embeddings = []
        total = len(documents)
        BATCH_SIZE = 20

        for i in range(0, total, BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            print(f"Embedding batch {i} -> {i + len(batch)}")

            # THIS LOOP REPLACES THE DECORATOR
            while True:  
                try:
                    # Try to get data
                    batch_embeddings = self.embed_batch(batch)
                    embeddings.extend(batch_embeddings)
                    break  # Success! Break the retry loop

                except Exception as e:
                    error_msg = str(e)
                    
                    # CASE 1: SPEED LIMIT (Sleep 65s)
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        print(f"QUOTA EXCEEDED! Sleeping for 65 seconds...")
                        time.sleep(65.0)
                        continue # Retry the loop
                    
                    # CASE 2: SERVER ERROR (Sleep 5s)
                    elif "503" in error_msg or "500" in error_msg:
                        print(f"Server hiccup. Sleeping 5s...")
                        time.sleep(5.0)
                        continue # Retry the loop

                    # CASE 3: BAD DATA (Don't retry, just skip)
                    else:
                        print(f"Batch failed with error: {e}. Switching to One-by-One Fallback.")
                        
                        # Iterate through the problematic batch, one document at a time
                        for doc in batch:
                            try:
                                # Try to embed just this one document
                                single_emb = self.embed_batch([doc])
                                embeddings.extend(single_emb)
                                
                                # CRITICAL: Sleep between single docs to avoid hitting rate limits
                                time.sleep(5.0) 
                                
                            except Exception as inner_e:
                                # If this specific doc fails, log it and append None.
                                # This keeps the list length aligned with the metadata.
                                print(f"PERMANENT FAIL: Skipping malformed doc: {inner_e}")
                                embeddings.append(None)
                        
                        # We have manually processed this batch, so we break the retry loop
                        break

            time.sleep(2.0) # Politeness sleep

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query],
            config=types.EmbedContentConfig(
                task_type="retrieval_query"
            )
        )
        return response.embeddings[0].values
