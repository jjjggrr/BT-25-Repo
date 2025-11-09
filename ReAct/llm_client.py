# llm_client.py
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time

load_dotenv()


def _safe_call(fn, retries=3, delay=1.0):
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            print(f"[GeminiClient] Attempt {i + 1}/{retries} failed: {e}")
            time.sleep(delay)
    return None


class GeminiClient:
    def __init__(self, model="gemini-2.5-flash"):
        # Neue Syntax: API-Key wird direkt beim Client-Objekt übergeben
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model

    def generate_answer(self, prompt: str):
        timings = {}
        answer_text = None

        try:
            # --- API Call ---
            t_api_start = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=3000,
                    temperature=0.2,
                ),
            )
            timings["t_llm_api_2"] = time.time() - t_api_start

            # --- Nachbearbeitung / Parsing (Overhead) ---
            t_post_start = time.time()
            if not response or not getattr(response, "candidates", None):
                print("[GeminiClient] No candidates returned for answer.")
                return None, timings
            parts = response.candidates[0].content.parts
            if not parts:
                print("[GeminiClient] No content parts in answer.")
                return None, timings
            text = parts[0].text if hasattr(parts[0], "text") else None
            if text:
                answer_text = text.strip()
            else:
                print("[GeminiClient] No text in answer output.")
            timings["t_llm_postprocess_answer"] = time.time() - t_post_start

            print(f"[GeminiClient] Answer generated ({len(answer_text or '')} chars).")
            return answer_text, timings

        except Exception as e:
            print(f"[GeminiClient] Error generating answer: {e}")
            return None, timings

    def generate_queries(self, question: str, schema: dict) -> list[dict]:
        import json

        schema_text = json.dumps(schema, indent=2)
        prompt = f"""
        You are a Cube.js query generator.
        Given this schema and its valid values, create one or more **independent JSON objects**,
        each answering part of the question: '{question}'.

        Rules:
        - Output ONLY raw JSON objects.
        - Do NOT include markdown fences like ```json.
        - Separate each object with a single pipe character: |
        - Ensure every JSON object is syntactically complete and closed.
        Schema:
        {schema_text}
        """

        timings = {}
        queries = []

        try:
            # --- API Call ---
            t_api_start = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=3500,
                    temperature=0.3,
                ),
            )
            timings["t_llm_api_1"] = time.time() - t_api_start

            # --- Prüfen & Text extrahieren ---
            if not response or not getattr(response, "candidates", None):
                print("[GeminiClient] No candidates returned.")
                return [], timings
            parts = response.candidates[0].content.parts
            if not parts:
                print("[GeminiClient] Empty Gemini response parts.")
                return [], timings
            text = parts[0].text if hasattr(parts[0], "text") else None
            if not text:
                print("[GeminiClient] No textual output from Gemini.")
                return [], timings

            print("\n=== GEMINI RAW QUERY OUTPUT ===")
            print(text[:1000])
            print("=== END RAW QUERY OUTPUT ===\n")

            # --- Parsing (Overhead nach API) ---
            t_parse_start = time.time()
            for i, part in enumerate(text.split("|"), start=1):
                part = part.strip()
                if not part:
                    continue
                part_fixed = part
                if part_fixed.count("{") > part_fixed.count("}"):
                    part_fixed += "}" * (part_fixed.count("{") - part_fixed.count("}"))
                try:
                    if part.startswith("[") and part.endswith("]"):
                        parsed = json.loads(part)
                        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                            q = parsed[0]
                        else:
                            q = parsed
                    else:
                        q = json.loads(part)
                    queries.append(q)
                    print(f"[GeminiClient] Parsed Query #{i}: {json.dumps(q, indent=2)[:300]}...")
                except json.JSONDecodeError:
                    print(f"[GeminiClient] ⚠️ Skipped malformed query fragment #{i}")
            timings["t_llm_parse_queries"] = time.time() - t_parse_start

            print(f"[GeminiClient] → Parsed {len(queries)} valid queries.\n")
            return queries, timings

        except Exception as e:
            print(f"[GeminiClient] Error during query generation: {e}")
            return [], timings

