# utils/merge_utils.py

import json

def merge_entities(gpt_entities_json: str, spacy_entities: list):
    try:
        gpt_entities = json.loads(gpt_entities_json)
    except:
        gpt_entities = []

    merged = { (e["entity"].lower(), e["type"]): e for e in gpt_entities }

    for ent in spacy_entities:
        key = (ent["entity"].lower(), ent["type"])
        if key not in merged:
            merged[key] = ent

    return list(merged.values())

def merge_relations(gpt_rel_json: str, spacy_rel: list):
    try:
        gpt_rel = json.loads(gpt_rel_json)
    except:
        gpt_rel = []

    merged = { (r["source"], r["target"], r["relation"]): r for r in gpt_rel }

    for rel in spacy_rel:
        key = (rel["source"], rel["target"], rel["relation"])
        if key not in merged:
            merged[key] = rel

    return list(merged.values())
