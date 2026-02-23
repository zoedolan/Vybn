"""Model adapters for local inference backends."""
from typing import Callable


def llama_cpp_adapter(endpoint: str = "http://localhost:8080",
                      temperature: float = 0.1) -> Callable[[str], str]:
    import httpx
    def call(prompt: str) -> str:
        resp = httpx.post(f"{endpoint}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}],
                  "temperature": temperature, "max_tokens": 4096}, timeout=120)
        return resp.json()["choices"][0]["message"]["content"]
    return call


def ollama_adapter(model: str = "minimax-m2.5",
                   endpoint: str = "http://localhost:11434") -> Callable[[str], str]:
    import httpx
    def call(prompt: str) -> str:
        resp = httpx.post(f"{endpoint}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
        return resp.json()["response"]
    return call
