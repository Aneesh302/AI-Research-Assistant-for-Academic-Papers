import time
import json
import boto3
from typing import List
from botocore.config import Config
from config import BEDROCK_REGION, EMBEDDING_MODEL_ID, BATCH_SIZE

class BedrockEmbedder:
    def __init__(self):
        # Retry config handles occasional AWS throttling automatically
        config = Config(
            retries = dict(
                max_attempts = 8
            )
        )
        # Initialize the AWS Bedrock client
        self.client = boto3.client(
            "bedrock-runtime", 
            region_name=BEDROCK_REGION, 
            config=config
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        # Bedrock Titan can accept arrays, but for safety and error handling
        # we process them in a tight loop here. It's extremely fast (~20ms per doc).
        for text in texts:
            try:
                # 1. Prepare Payload
                body = json.dumps({
                    "inputText": text,
                    "dimensions": 1024, # Ensure we strictly request 1024
                    "normalize": True   # Good for cosine similarity
                })
                
                # 2. Call API
                response = self.client.invoke_model(
                    modelId=EMBEDDING_MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=body
                )
                
                # 3. Parse Response
                response_body = json.loads(response.get("body").read())
                embedding = response_body.get("embedding")
                embeddings.append(embedding)

            except Exception as e:
                print(f"Error embedding document: {e}")
                embeddings.append(None)
                
        return embeddings

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        embeddings = []
        total = len(documents)

        print(f"Starting AWS Bedrock Embedding for {total} documents...")

        for i in range(0, total, BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            print(f"Embedding batch {i} -> {i + len(batch)}")

            try:
                batch_embeddings = self.embed_batch(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Batch failed: {e}")
                # Fill with None to keep alignment
                embeddings.extend([None] * len(batch))

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        # Helper for a single query
        result = self.embed_batch([query])
        return result[0] if result else []