# llm_client.py
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class GeminiClient:
    def __init__(self, model="gemini-2.5-flash"):
        # Neue Syntax: API-Key wird direkt beim Client-Objekt übergeben
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model

    def generate_answer(self, prompt: str, context: str = None) -> str:
        try:
            full_prompt = f"{prompt}\n\nContext:\n{context or ''}"
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=2048,
                    temperature=0.4
                ),
            )

            if not response or not getattr(response, "candidates", None):
                return "No content returned."

            parts = response.candidates[0].content.parts
            text = "\n".join(p.text for p in parts if hasattr(p, "text"))
            return text.strip() if text else "No text output."
        except Exception as e:
            print(f"[GeminiClient] Error during generation: {e}")
            return None

    def generate_queries(self, question: str, schema: dict) -> list[dict]:
        import json

        schema_text = json.dumps(schema, indent=2)
        prompt = f"""
You are a Cube.js query generator.
Given this schema and its valid values, create one or more JSON queries
to answer the question: '{question}'.

Rules:
- Each query must be valid JSON.
- If a comparison (like FY24 vs FY25) is required, output multiple queries separated by '|'.
- Do NOT explain anything, only output queries.
Schema:
{schema_text}
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=2048,
                    temperature=0.3
                ),
            )

            # Safety: empty or missing candidates
            if not response or not getattr(response, "candidates", None):
                print("[GeminiClient] No candidates returned from model.")
                return []

            # Extract raw text
            parts = response.candidates[0].content.parts
            if not parts:
                print("[GeminiClient] No content parts found in response.")
                return []

            text = parts[0].text if hasattr(parts[0], "text") else None
            if not text:
                print("[GeminiClient] No textual output from Gemini.")
                return []

            print("\n=== GEMINI RAW QUERY OUTPUT ===")
            print(text)
            print("=== END RAW QUERY OUTPUT ===\n")

            queries = []
            for i, part in enumerate(text.split("|"), start=1):
                part = part.strip()
                if not part:
                    continue
                try:
                    q = json.loads(part)
                    queries.append(q)
                    print(f"[GeminiClient] Parsed Query #{i}: {json.dumps(q, indent=2)[:300]}...")
                except json.JSONDecodeError:
                    print(f"[GeminiClient] ⚠️ Skipped malformed query fragment #{i}: {part[:160]}")

            print(f"[GeminiClient] → Parsed {len(queries)} valid queries.\n")
            return queries

        except Exception as e:
            print(f"[GeminiClient] Error generating queries: {e}")
            return []
