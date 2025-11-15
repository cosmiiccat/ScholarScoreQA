# graphrag_core/entity_extractor.py

from .config import GraphRAGConfig

ENTITY_PROMPT = """
You are an information extraction assistant.
Extract all entities from the text below and classify them into meaningful types:
- PERSON
- ORGANIZATION
- LOCATION
- DATE
- PRODUCT
- EVENT
- CONCEPT
- OTHER

Be thorough but concise.
Return valid JSON list:

[
  { "entity": "EntityName", "type": "EntityType" }
]

Text:
\"\"\"{context}\"\"\"
"""

class EntityExtractor:
    def __init__(self, client):
        self.client = client

    def extract(self, context: str) -> str:
        prompt = ENTITY_PROMPT.format(context=context)
        return self.client.run(prompt, GraphRAGConfig.MODEL_NAME)
