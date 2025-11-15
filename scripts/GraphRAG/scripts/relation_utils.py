# utils/relation_utils.py

def format_relations(rel_list: list):
    return sorted(rel_list, key=lambda x: (x["source"], x["relation"], x["target"]))
