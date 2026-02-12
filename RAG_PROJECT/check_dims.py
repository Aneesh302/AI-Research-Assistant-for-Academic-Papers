import os
from google import genai
from google.genai import types

# 1. Setup
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL_NAME = "gemini-embedding-001"

# 2. Embed a single word
response = client.models.embed_content(
    model=MODEL_NAME,
    contents=["Test"],
    config=types.EmbedContentConfig(task_type="retrieval_document")
)

# 3. Print the Truth
vector = response.embeddings[0].values
print(f"Model: {MODEL_NAME}")
print(f"True Dimensions: {len(vector)}")