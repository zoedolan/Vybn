import os
import requests


class _Completions:
    """Minimal wrapper around the OpenAI chat completion endpoint."""

    API_URL = "https://api.openai.com/v1/chat/completions"

    def create(self, model: str, messages: list[dict], user: str, timeout: int = 30, **kwargs):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "user": user}
        resp = requests.post(self.API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]["content"]
        return type("Resp", (), {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": choice})()})()]})

class chat:
    completions = _Completions()

api_key = None
