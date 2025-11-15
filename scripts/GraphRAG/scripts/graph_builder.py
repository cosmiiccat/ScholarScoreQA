# graphrag_core/graph_builder.py

import json

class KnowledgeGraph:
    """
    Simple in-memory knowledge graph constructed from extracted entities + relations.
    """

    def __init__(self):
        self.entities = []
        self.relations = []

    def load_entities(self, entity_json: str):
        try:
            parsed = json.loads(entity_json)
            self.entities = parsed
        except Exception:
            self.entities = []

    def load_relations(self, relation_json: str):
        try:
            parsed = json.loads(relation_json)
            self.relations = parsed
        except Exception:
            self.relations = []

    def summary_text(self):
        """
        Convert graph to a textual summary that can be fed into a QA model.
        """
        lines = ["Entities:"]
        for e in self.entities:
            lines.append(f"- {e['entity']} ({e['type']})")

        lines.append("\nRelations:")
        for r in self.relations:
            lines.append(f"- {r['source']} --[{r['relation']}]--> {r['target']}")

        return "\n".join(lines)
