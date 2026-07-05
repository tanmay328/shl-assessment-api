# app/llm/generator.py - Using HuggingFace Inference API (No transformers)
import requests
import os
from typing import Optional

class LLMGenerator:
    def __init__(self):
        """Initialize using free HuggingFace Inference API"""
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/phi-2"
        self.headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN', '')}"}
        self.loaded = False
        
        # Check if we have a token
        if os.getenv('HUGGINGFACE_TOKEN'):
            self.loaded = True
            print("Using HuggingFace Inference API (Free)")
        else:
            print("No HuggingFace token found. Using fallback mode.")
            print("To get a free token: https://huggingface.co/settings/tokens")
    
    def generate_response(self, prompt: str, max_length: int = 512) -> Optional[str]:
        """Generate response using HuggingFace Inference API.

        Timeout kept short (8s) so a slow/unavailable free-tier model can't
        eat the evaluator's 30-second-per-call budget; callers should treat
        a None return as "fall back to the template response".
        """
        if not self.loaded:
            return None
        
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "do_sample": True
                }
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=8)
            
            if response.status_code == 200:
                result = response.json()
                if result and isinstance(result, list) and result[0].get('generated_text'):
                    generated = result[0]['generated_text']
                    # Remove the prompt from the response
                    return generated.replace(prompt, '').strip()
            elif response.status_code == 503:
                # Model is loading — don't wait, just fall back this turn
                print("HF model is loading (503); falling back to template response")
                return None
            else:
                print(f"API error: {response.status_code}")
            return None
        except requests.exceptions.Timeout:
            print("HF API timed out; falling back to template response")
            return None
        except Exception as e:
            print(f"API error: {e}")
            return None