# tone_judge/tone_scorer.py

"""
Tone Scoring:
- Tone Fidelity (LLM-based)
- Content Preservation (semantic similarity)
- Linguistic Clarity (readability)
"""

from typing import Tuple
from sentence_transformers import SentenceTransformer, util
from .readability import Readability
from .api_client import APIClient
from .config import OPENAI_MODEL


class ToneScorer:
    def __init__(self):
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.client = APIClient()

    # ------------------------------------------------------------
    def score_tone_fidelity(self, transformed: str, requested_tone: str) -> float:
        """
        Use LLM to score tone fidelity (0–1).
        """
        prompt = f"""
Rate the tone fidelity (0 to 1) of the following answer.

Requested Tone: {requested_tone}

Answer:
\"\"\"{transformed}\"\"\"
Only return a number between 0 and 1.
"""

        resp = self.client.chat(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        try:
            return float(resp.choices[0].message.content.strip())
        except:
            return 0.0

    # ------------------------------------------------------------
    def score_content_preservation(self, original: str, transformed: str) -> float:
        """Semantic similarity using Sentence Transformers."""
        e1 = self.encoder.encode(original, convert_to_tensor=True)
        e2 = self.encoder.encode(transformed, convert_to_tensor=True)
        sim = util.cos_sim(e1, e2).item()
        return max(0.0, min(sim, 1.0))

    # ------------------------------------------------------------
    def score_linguistic_clarity(self, transformed: str) -> float:
        """Use readability metrics."""
        return Readability.clarity_score(transformed)
