# graphrag_core/graph_rag.py

from .entity_extractor import EntityExtractor
from .relation_extractor import RelationExtractor
from .graph_builder import KnowledgeGraph
from .config import GraphRAGConfig

QA_PROMPT = """
You are a GraphRAG question-answering assistant.

Use:
1. The original context
2. The structured knowledge graph summary

Context:
\"\"\"{context}\"\"\"

Knowledge Graph:
\"\"\"{graph}\"\"\"

Question:
{question}

If insufficient information exists, respond:
"The provided context does not contain enough information."

Return a short, direct answer.
"""

class GraphRAGPipeline:
    def __init__(self, client):
        self.client = client
        self.entity_extractor = EntityExtractor(client)
        self.relation_extractor = RelationExtractor(client)

    def run(self, context: str, question: str):
        # ---- Step 1: Extract Entities ----
        entities = self.entity_extractor.extract(context)

        # ---- Step 2: Extract Relations ----
        relations = self.relation_extractor.extract(context, entities)

        # ---- Step 3: Build Knowledge Graph ----
        graph = KnowledgeGraph()
        graph.load_entities(entities)
        graph.load_relations(relations)

        graph_summary = graph.summary_text()

        # ---- Step 4: Graph-aware QA ----
        final_prompt = QA_PROMPT.format(
            context=context,
            graph=graph_summary,
            question=question
        )
        answer = self.client.run(final_prompt, GraphRAGConfig.MODEL_NAME)

        return {
            "entities": entities,
            "relations": relations,
            "graph_summary": graph_summary,
            "answer": answer
        }
