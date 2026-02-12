# scripts/ingest_abstracts.py
import time
import pandas as pd

# REMOVED: Google imports and API Key checks are no longer needed
from embeddings.embedder import BedrockEmbedder
from storage.abstract_store import AbstractVectorStore


# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------
FILE_PATH = "./data/arxiv-metadata-oai-snapshot.json"
CHUNK_SIZE = 10_000
START_DATE = "2025-01-01"
REQUIRED_CATEGORIES = {"cs.AI", "cs.LG"}


# ------------------------------------------------
# INITIALIZE SERVICES
# ------------------------------------------------
# NOTE: No API Key needed here. BedrockEmbedder uses your local AWS credentials
embedder = BedrockEmbedder()
abstract_store = AbstractVectorStore()


# ------------------------------------------------
# INGESTION LOGIC
# ------------------------------------------------
def main():
    start_time = time.time()
    total_embedded = 0

    print("Starting abstract ingestion (AWS Bedrock Titan V2)")
    print(f"Start date filter: {START_DATE}")
    print(f"Required categories: {REQUIRED_CATEGORIES}")
    print(f"Chunk size: {CHUNK_SIZE}\n")

    chunks = pd.read_json(
        FILE_PATH,
        lines=True,
        chunksize=CHUNK_SIZE
    )

    for chunk_idx, chunk in enumerate(chunks, start=1):
        print(f"\nProcessing chunk {chunk_idx}")

        # -----------------------------
        # FILTER BY CATEGORY + DATE
        # -----------------------------
        # Filter Logic: Must have BOTH categories and match the date
        filtered = chunk[
            (chunk["categories"].apply(
                lambda cat: REQUIRED_CATEGORIES.issubset(set(cat.split()))
            )) &
            (pd.to_datetime(chunk["update_date"], errors="coerce")
             >= pd.Timestamp(START_DATE))
        ]

        if filtered.empty:
            print("No matching papers in this chunk")
            continue

        # -----------------------------
        # PREPARE DOCUMENTS + METADATA
        # -----------------------------
        documents = []
        records = []

        for _, row in filtered.iterrows():
            # Construct the text to be embedded (Title + Abstract)
            documents.append(
                f"{row['title']}\n\n{row['abstract']}"
            )

            records.append({
                "arxiv_id": row["id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "categories": row["categories"],
                "update_date": row["update_date"]
            })

        print(f"Embedding {len(documents)} abstracts...")

        # -----------------------------
        # EMBED & STORE
        # -----------------------------
        # 1. Get Embeddings (Bedrock Titan V2)
        # Note: Bedrock handles retries internally via the Embedder class
        embeddings = embedder.embed_documents(documents)
        
        # 2. Filter out failures (None values) BEFORE saving
        valid_records = []
        valid_embeddings = []
        
        for record, emb in zip(records, embeddings):
            if emb is not None:
                valid_records.append(record)
                valid_embeddings.append(emb)
            else:
                print(f"Dropping failed paper: {record['title']}")

        # 3. Store only the valid pairs
        if valid_records:
            abstract_store.add_documents(valid_records, valid_embeddings)
            total_embedded += len(valid_records)
            print(f"Stored {len(valid_records)} papers (dropped {len(records) - len(valid_records)})")

    # ------------------------------------------------
    # FINAL LOG
    # ------------------------------------------------
    end_time = time.time()
    elapsed = end_time - start_time

    print("\nAbstract ingestion complete")
    print(f"Total papers embedded: {total_embedded}")
    print(f"Total time: {elapsed / 60:.2f} minutes")


# ------------------------------------------------
# ENTRY POINT
# ------------------------------------------------
if __name__ == "__main__":
    main()