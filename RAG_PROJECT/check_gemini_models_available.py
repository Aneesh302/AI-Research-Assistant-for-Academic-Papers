from google import genai
from config import GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)

print("List of all available models:")
for m in client.models.list():
    print(m.name)