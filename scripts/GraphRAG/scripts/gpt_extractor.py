# models/gpt_extractor.py

from config import OPENAI_CLIENT, GPT_MODEL

class GPTExtractor:

    @staticmethod
    def extract_entities(text: str) -> list:
        prompt = f"""
Extract ALL entities from the text. Provide JSON list:
[
  {{ "entity": "...", "type": "..." }}
]
Text:
\"\"\"{text}\"\"\"
"""
        resp = OPENAI_CLIENT.responses.create(model=GPT_MODEL, input=prompt)
        return resp.output[0].content[0].text

    @staticmethod
    def extract_relations(text: str, entities_json: str) -> list:
        prompt = f"""
Identify ALL relations between entities.

Text:
\"\"\"{text}\"\"\"
Entities:
{entities_json}

Return JSON list:
[
  {{
    "source": "...",
    "target": "...",
    "relation": "..."
  }}
]
"""
        resp = OPENAI_CLIENT.responses.create(model=GPT_MODEL, input=prompt)
        return resp.output[0].content[0].text

    @staticmethod
    def answer_question(context: str, question: str) -> str:
        prompt = f"""
Answer the question from context. If insufficient info, say so.

Context:
\"\"\"{context}\"\"\"
Question: {question}
"""
        resp = OPENAI_CLIENT.responses.create(model=GPT_MODEL, input=prompt)
        return resp.output[0].content[0].text
