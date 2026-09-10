import os
import cv2
import json
import base64
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load .env variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

class AIGlyphValidator:
    """
    Multimodal AI Vision Agent that inspects candidate character segmentation crops
    from historical Modi manuscripts using local Ollama models (e.g. gemma4:12b) or Gemini API.
    """

    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider or os.getenv("AI_VALIDATOR_PROVIDER", "ollama").lower()
        self.model_name = model_name or os.getenv("AI_VALIDATOR_MODEL", "gemma4:12b")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._gemini_client = None

        if self.provider == "gemini":
            self._init_gemini_client()

    def _init_gemini_client(self):
        """Initializes the Gemini client SDK if provider is gemini."""
        if not self.api_key or "your_" in self.api_key:
            return

        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=self.api_key)
        except Exception:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self._gemini_client = legacy_genai.GenerativeModel(self.model_name)
            except Exception as e:
                print(f"[AIGlyphValidator] Warning: Could not initialize Gemini SDK: {e}")

    @staticmethod
    def _img_to_base64_png(img: np.ndarray) -> str:
        """Converts an OpenCV image to base64 encoded PNG string."""
        if len(img.shape) == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            bgr = img
        _, buffer = cv2.imencode(".png", bgr)
        return base64.b64encode(buffer).decode("utf-8")

    def _call_ollama(self, prompt: str, base64_images: List[str]) -> str:
        """Sends multimodal request to local Ollama server."""
        url = f"{self.ollama_host}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": base64_images
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        res_json = resp.json()
        return res_json.get("message", {}).get("content", "")

    def validate_single_glyph(
        self,
        annotated_line_img: np.ndarray,
        target_glyph_crop: np.ndarray,
        glyph_idx: int,
        total_glyphs: int
    ) -> Dict[str, Any]:
        """
        Inspects an individual candidate glyph chosen by the user in the GUI.
        Returns a structured action: ACCEPT, MERGE_LEFT, MERGE_RIGHT, or SPLIT_AT_VALLEY.
        """
        system_instruction = (
            "You are an expert Document Analysis and Paleography Vision Agent specializing in archaic Modi script (मोडी लिपी).\n"
            "The user selected candidate glyph #" + str(glyph_idx + 1) + " (out of " + str(total_glyphs) + ") from the text line for verification.\n\n"
            "### MODI SCRIPT STRUCTURAL RULES:\n"
            "1. COMPLETE SYLLABLE (Akshara): A valid Modi character contains the base consonant loop/body + internal stems + trailing Kana (ा) + top/bottom matras.\n"
            "2. NEVER AN ORPHAN KANA: A character NEVER consists of an isolated vertical bar (Kana 'ा') or single stem. If this crop is only a trailing vertical stroke or broken tail, choose MERGE_LEFT.\n"
            "3. MULTI-STEM CONSONANTS ('शा', 'ळा', 'गाडा', 'ता', 'मा'): Slicing between the consonant body and its trailing kana stem is an error -> MERGE_LEFT.\n"
            "4. TOUCHING CLUSTER: If this crop contains 2 touching characters connected at the baseline, choose SPLIT_AT_VALLEY.\n\n"
            "### REQUIRED OUTPUT FORMAT (JSON ONLY):\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "glyph_id": ' + str(glyph_idx + 1) + ',\n'
            '  "action": "ACCEPT" | "MERGE_LEFT" | "MERGE_RIGHT" | "SPLIT_AT_VALLEY",\n'
            '  "reason": "short explanation",\n'
            '  "split_ratio": 0.5\n'
            "}"
        )

        user_prompt = (
            f"Inspect Glyph #{glyph_idx + 1}.\n"
            "Image 1 is the full line overview.\n"
            "Image 2 is the isolated candidate glyph crop.\n"
            "Decide the correction action and return JSON."
        )

        b64_line = self._img_to_base64_png(annotated_line_img)
        b64_glyph = self._img_to_base64_png(target_glyph_crop)

        # 1. Ollama Provider
        if self.provider == "ollama":
            try:
                raw_text = self._call_ollama(f"{system_instruction}\n\n{user_prompt}", [b64_line, b64_glyph])
                return self._parse_json_response(raw_text)
            except Exception as e:
                print(f"[AIGlyphValidator:Ollama] Error inspecting glyph {glyph_idx+1}: {e}")
                return {"status": "error", "error": str(e), "action": "ACCEPT"}

        # 2. Gemini Provider
        elif self._gemini_client:
            try:
                from PIL import Image
                import io
                pil_line = Image.fromarray(cv2.cvtColor(annotated_line_img, cv2.COLOR_BGR2RGB))
                pil_glyph = Image.fromarray(cv2.cvtColor(target_glyph_crop, cv2.COLOR_GRAY2RGB))
                contents = [system_instruction, user_prompt, pil_line, pil_glyph]

                if hasattr(self._gemini_client, "models"):
                    response = self._gemini_client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config={"response_mime_type": "application/json"}
                    )
                    raw_text = response.text
                else:
                    response = self._gemini_client.generate_content(contents)
                    raw_text = response.text

                return self._parse_json_response(raw_text)
            except Exception as e:
                print(f"[AIGlyphValidator:Gemini] Error inspecting glyph {glyph_idx+1}: {e}")
                return {"status": "error", "error": str(e), "action": "ACCEPT"}

        return {"status": "fallback", "action": "ACCEPT"}

    def validate_line_segmentation(
        self,
        annotated_line_img: np.ndarray,
        glyph_crops: List[np.ndarray],
        line_id: int = 0
    ) -> Dict[str, Any]:
        """
        Submits candidate character crops along with the annotated line overview to the Vision Model.
        """
        if not glyph_crops:
            return {"status": "fallback", "verdicts": []}

        system_instruction = (
            "You are an expert Document Analysis and Paleography Vision Agent specializing in archaic Modi script (मोडी लिपी).\n"
            "Inspect candidate crops extracted from a line and evaluate Akshara completeness.\n"
            "RULES:\n"
            "1. COMPLETE SYLLABLE: Base loop + internal stems + trailing Kana (ा) + matras.\n"
            "2. NO ORPHAN KANA: If a crop is just an isolated stem/bar -> MERGE_LEFT.\n"
            "3. MULTI-STEM CONSONANTS ('शा', 'ळा', 'गा', 'डा') -> MERGE_LEFT.\n"
            "4. TOUCHING CLUSTERS -> SPLIT_AT_VALLEY.\n\n"
            "OUTPUT FORMAT (Strict JSON Array):\n"
            "[\n"
            '  {"glyph_id": 1, "action": "ACCEPT", "reason": "complete character"},\n'
            '  {"glyph_id": 2, "action": "MERGE_LEFT", "reason": "orphan kana stem"}\n'
            "]"
        )

        user_prompt = f"Line #{line_id + 1} with {len(glyph_crops)} glyph crops. Provide your JSON verdict array."

        # 1. Ollama Provider
        if self.provider == "ollama":
            try:
                images_b64 = [self._img_to_base64_png(annotated_line_img)] + [self._img_to_base64_png(c) for c in glyph_crops]
                raw_text = self._call_ollama(f"{system_instruction}\n\n{user_prompt}", images_b64)
                parsed = self._parse_json_response(raw_text)
                return {"status": "success", "data": parsed}
            except Exception as e:
                print(f"[AIGlyphValidator:Ollama] Line {line_id+1} Error: {e}")
                return {"status": "error", "error": str(e), "verdicts": []}

        # 2. Gemini Provider
        elif self._gemini_client:
            try:
                from PIL import Image
                pil_images = [Image.fromarray(cv2.cvtColor(annotated_line_img, cv2.COLOR_BGR2RGB))]
                for c in glyph_crops:
                    pil_images.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_GRAY2RGB)))

                contents = [system_instruction, user_prompt] + pil_images
                if hasattr(self._gemini_client, "models"):
                    response = self._gemini_client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config={"response_mime_type": "application/json"}
                    )
                    raw_text = response.text
                else:
                    response = self._gemini_client.generate_content(contents)
                    raw_text = response.text

                parsed = self._parse_json_response(raw_text)
                return {"status": "success", "data": parsed}
            except Exception as e:
                print(f"[AIGlyphValidator:Gemini] Line {line_id+1} Error: {e}")
                return {"status": "error", "error": str(e), "verdicts": []}

        return {"status": "fallback", "verdicts": []}

    @staticmethod
    def _parse_json_response(raw_text: str) -> Any:
        """Robust JSON extraction from LLM text."""
        cleaned = raw_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        # Find first '{' or '['
        first_brace = cleaned.find("{")
        first_bracket = cleaned.find("[")
        
        start = 0
        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
            start = first_brace
            end = cleaned.rfind("}") + 1
            cleaned = cleaned[start:end]
        elif first_bracket != -1:
            start = first_bracket
            end = cleaned.rfind("]") + 1
            cleaned = cleaned[start:end]

        return json.loads(cleaned)
