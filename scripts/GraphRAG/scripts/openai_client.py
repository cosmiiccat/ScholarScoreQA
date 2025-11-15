# graphrag_core/openai_client.py

from openai import OpenAI

class OpenAIClientWrapper:
    """
    A small wrapper for OpenAI responses, so the rest of the code is cleaner.
    """

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def run(self, prompt: str, model: str):
        response = self.client.responses.create(
            model=model,
            input=prompt
        )
        return response.output_text
