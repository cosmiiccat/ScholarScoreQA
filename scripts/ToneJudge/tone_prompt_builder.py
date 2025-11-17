# tone_judge/tone_prompt_builder.py

"""
Build structured prompts for tone transformation.
"""

from typing import Dict
from .config import VALID_TONES


class TonePromptBuilder:

    @staticmethod
    def build_prompt(answer: str, tone_key: str) -> str:
        if tone_key not in VALID_TONES:
            raise ValueError(f"Invalid tone: {tone_key}. Choose from {list(VALID_TONES.keys())}")

        tone_desc = VALID_TONES[tone_key]

        return f"""
You are an expert in adapting writing tone to different audiences.

Tone Definitions:
1. Academic – formal, precise, scholarly.
2. Technical – concise, factual, uses domain-specific vocabulary.
3. Descriptive – explanatory, illustrative.
4. Conversational – casual, friendly, natural.
5. Simplified – easy language, no jargon.

Original Answer:
\"\"\"{answer}\"\"\"

Requested Tone:
{tone_desc}

Rewrite the answer in the specified tone while preserving meaning.
Return ONLY the rewritten answer.
"""
