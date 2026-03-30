from openai import OpenAI
from typing import List, Dict, Generator


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages: List[Dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2
        )
        return resp.choices[0].message.content or ""

    def chat_stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta