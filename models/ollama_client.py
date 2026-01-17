import requests
from typing import Dict, Any


class OllamaClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.base_url = "http://localhost:11434/api/generate"

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }

        response = requests.post(self.base_url, json=payload, timeout=120)
        response.raise_for_status()

        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "output": response.json().get("response", "").strip(),
        }
