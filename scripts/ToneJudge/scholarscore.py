# scholar_score.py

class ScholarScore:
    """
    Computes ScholarScore(a) as defined in the paper:

    ScholarScore(a) = 2 * LJ(a) * TJ(a) / (LJ(a) + TJ(a))

    LJ(a) = Language Judge final score  (0–1)
    TJ(a) = Tone Judge final score      (0–1)
    """

    @staticmethod
    def compute(language_judge_score: float, tone_judge_score: float) -> float:
        lj = language_judge_score
        tj = tone_judge_score

        # Avoid division by zero
        if lj == 0 and tj == 0:
            return 0.0

        return (2 * lj * tj) / (lj + tj)