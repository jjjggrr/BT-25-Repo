import os
from typing import Dict, Any
from google import genai
from google.genai import types


class GeminiClient:
    def __init__(self, model: str = "gemini-2.5-flash", api_key: str = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable GEMINI_API_KEY nicht gesetzt.")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate_answer(
        self,
        prompt: str,
        context: Dict[str, Any],
        max_output_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """
        Sendet den Prompt + Kontext an das Modell und gibt eine Textantwort zurück.
        """
        context_text = f"Context:\n{context}"
        question_text = f"Question:\n{prompt}"

        contents = types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=context_text),
                types.Part.from_text(text=question_text)
            ]
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            ),
        )
        print("=== GEMINI RAW RESPONSE ===")
        print(response)
        print("============================")

        # --- Sicher extrahieren ---
        if hasattr(response, "text") and response.text:
            return response.text.strip()

        # Fallback: Suche Text in "output" oder "candidates"
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts:
                text_parts = [p.text for p in parts if hasattr(p, "text")]
                return "\n".join(text_parts).strip() if text_parts else "No text content."

        return "No content returned."
