# language_judge/scorers/format_scorer.py

class FormatScorer:
    """
    Format Compliance:
    Rule-based + LLM could be added.
    """

    @staticmethod
    def score(answer: str) -> float:
        # Basic heuristic:
        if len(answer.strip().split(".")) < 1:
            return 0.2  # poorly formatted

        if answer and answer[0].isupper() and answer[-1] == ".":
            return 1.0

        return 0.7
