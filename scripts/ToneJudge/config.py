# tone_judge/config.py

"""
Configuration for Tone Transformer & Tone Judge Module.
"""

TONE_WEIGHT_TFIDELITY = 0.40
TONE_WEIGHT_CPRESERVE = 0.35
TONE_WEIGHT_CLARITY = 0.25

VALID_TONES = {
    "academic": "Academic (Scholars, researchers)",
    "technical": "Technical/Factual (Experts)",
    "descriptive": "Descriptive (General readers)",
    "conversational": "Conversational",
    "simplified": "Simplified/Layman-Friendly"
}

OPENAI_MODEL = "gpt-4o-mini"
