# rag/qa.py
from google.genai import types
from config import LLM_MODEL

class PDFQuestionAnsweringService:
    def __init__(self, embedder, pdf_store, genai_client):
        self.embedder = embedder
        self.store = pdf_store
        self.client = genai_client

    def answer(self, question: str, k: int = 5) -> str:
        query_vec = self.embedder.embed_query(question)
        contexts = self.store.search(query_vec, k=k)

        if not contexts:
            return "No relevant information found in the PDF."

        context_text = "\n\n".join(contexts)

        prompt = f"""
        You are an expert academic research assistant. Your goal is to provide a detailed, comprehensive, and well-structured answer based strictly on the provided context.

        Guidelines:
        - Do not be brief. Explain the concepts fully.
        - If the context describes a method, explain the steps in detail.
        - Use bullet points or numbered lists where appropriate to make it readable.
        - If the context mentions specific results or metrics, include them.
        - If the answer is not in the context, state that clearly.

        Context from the paper:
        {context_text}

        User Question: 
        {question}

        Detailed Answer:
        """

    
        # 3. Call the Model (Gemini 2.5 Flash / Flash Latest)
        model_candidates = [
            "gemini-2.5-flash",
            "gemini-flash-latest",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash"
        ]

        for model_name in model_candidates:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.3, # Low temp = more factual/focused
                        max_output_tokens=700 # Allow long answers
                    )
                )
                return response.text
                
            except Exception as e:
                print(f"Failed with {model_name}: {e}")
                continue 
        
        return "Error: Could not generate answer. All available models failed."
