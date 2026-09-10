import os
import json
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class VLMAgent:
    """
    Intelligent Agent that uses a Vision-Language Model via OpenRouter to visually inspect 
    a document and estimate the optimal classical CV parameters based on semantic context.
    """
    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def query_vlm_for_parameters(image_path, api_key=None):
        """
        Connects to OpenRouter to predict parameters using google/gemini-3-pro-image-preview.
        """
        # Prioritize the explicitly passed key, fallback to .env API_KEY
        actual_key = api_key if api_key else os.environ.get("API_KEY")
        
        # If no API key is provided, we return a smart default to prevent crashing
        if not actual_key:
            return {
                "x_stretch": 51,
                "y_stretch": 5,
                "savgol_window": 21,
                "ink_threshold": 5
            }
            
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=actual_key,
            )
            
            base64_image = VLMAgent.encode_image(image_path)
            
            prompt_text = """
            You are an expert Document Layout Analyst specializing in historical manuscripts.
            Look carefully at this document image. 
            Analyze the following properties:
            1. Are the text lines tightly packed together vertically, or are there wide gaps?
            2. Do characters from one line overlap with characters from another line?
            3. Are the individual letters in a word spaced far apart horizontally, or cursive and connected?
            
            Based on your visual analysis, predict the optimal parameters for an Anisotropic Gaussian Blob 
            segmentation algorithm and a Horizontal Projection Profile algorithm.
            
            Output ONLY a valid JSON object matching this schema exactly, nothing else:
            {
                "x_stretch": <integer between 21 and 81, higher if letters are spaced apart>,
                "y_stretch": <odd integer between 3 and 11, higher if line gaps are wide>,
                "savgol_window": <odd integer between 11 and 41, higher if lines are tall>,
                "ink_threshold": <integer between 3 and 15, higher if ink is very thick>
            }
            """
            
            response = client.chat.completions.create(
                model="google/gemini-3-pro-image-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            # Parse JSON
            raw_text = response.choices[0].message.content
            raw_json = raw_text.replace("```json", "").replace("```", "").strip()
            params = json.loads(raw_json)
            
            return params
            
        except Exception as e:
            print(f"VLM Agent Error: {e}")
            return None
