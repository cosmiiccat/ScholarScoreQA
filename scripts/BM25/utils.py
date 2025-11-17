# context_reducer/utils.py

"""
Utility methods for context reducer.
"""

from typing import List, Dict


def sort_by_key(data: List[Dict], key: str, reverse: bool = True) -> List[Dict]:
    """Sort list of dicts by key."""
    return sorted(data, key=lambda x: x[key], reverse=reverse)
