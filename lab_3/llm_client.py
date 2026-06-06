import json
import os
import re
from typing import Callable, Optional

import requests


def make_ollama_llm(
    model: str = "llama3.2",
    base_url: Optional[str] = None,
    temperature: float = 0.3,
) -> Callable[[str], str]:
    """Создаёт callable для вызова локальной модели через Ollama."""
    base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def llm_call(prompt: str) -> str:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    return llm_call


def extract_json(text: str) -> dict:
    """Извлекает JSON из ответа LLM (в т.ч. из markdown-блока)."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise
