# models/spacy_extractor.py

import spacy
from config import SPACY_NLP

class SpaCyExtractor:

    @staticmethod
    def extract_entities(text: str) -> list:
        doc = SPACY_NLP(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "entity": ent.text,
                "type": ent.label_
            })
        return entities

    @staticmethod
    def extract_relations(text: str):
        """
        Simple dependency-based relation detection.
        e.g., (PERSON) --nsubj--> (ORG)
        """
        doc = SPACY_NLP(text)
        relations = []

        for token in doc:
            if token.dep_ in ("nsubj", "dobj", "pobj") and token.head.pos_ in ("VERB", "NOUN"):
                source = token.text
                target = token.head.text
                relation = token.dep_

                relations.append({
                    "source": source,
                    "target": target,
                    "relation": relation
                })

        return relations
