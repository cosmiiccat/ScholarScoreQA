# graphrag_core/relation_extractor.py

from .config import GraphRAGConfig

RELATION_PROMPT = """
You are a relation extraction assistant.
Given the text and entity list, identify meaningful relations between them.

Each relation should describe a clear semantic link such as:
- founded by
- works at
- located in
- created
- owned by
- part of
- related to

Return JSON list:

[
  {
    "source": "Entity1",
    "target": "Entity2",
    "relation": "RelationType"
  }
]

Text:
\"\"\"{context}\"\"\"

Entities:
{entities}
"""

class RelationExtractor:
    def __init__(self, client):
        self.client = client

    def extract(self, context: str, entities: str) -> str:
        prompt = RELATION_PROMPT.format(context=context, entities=entities)
        return self.client.run(prompt, GraphRAGConfig.MODEL_NAME)
