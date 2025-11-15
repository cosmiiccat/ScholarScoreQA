# pipeline/entity_relation_builder.py

from .gpt_extractor import GPTExtractor
from .spacy_extractor import SpaCyExtractor
from .merge_utils import merge_entities, merge_relations

class EntityRelationBuilder:

    @staticmethod
    def run(text: str):
        # GPT extractions
        gpt_entities = GPTExtractor.extract_entities(text)
        gpt_relations = GPTExtractor.extract_relations(text, gpt_entities)

        # SpaCy extractions
        spacy_entities = SpaCyExtractor.extract_entities(text)
        spacy_relations = SpaCyExtractor.extract_relations(text)

        # Merging
        merged_entities = merge_entities(gpt_entities, spacy_entities)
        merged_relations = merge_relations(gpt_relations, spacy_relations)

        return {
            "entities": merged_entities,
            "relations": merged_relations
        }
