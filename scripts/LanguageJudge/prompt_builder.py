# language_judge/prompt_builder.py

class PromptBuilder:
    """Builds evaluation prompts for Language Judge."""

    @staticmethod
    def build(context: str, question: str, answer: str):
        return f"""
You are an expert evaluator. Score the ANSWER on 4 criteria in [0,1]:

1. Correctness: factual accuracy relative to CONTEXT and QUESTION.
2. Groundedness: strictly based on CONTEXT; no hallucination.
3. Fluency/Coherence: clarity and logical flow.
4. Format Compliance: follows instructions and expected structure.

Return ONLY a JSON:
{{
 "correctness": <score>,
 "groundedness": <score>,
 "fluency": <score>,
 "format": <score>
}}

CONTEXT:
\"\"\"{context}\"\"\"

QUESTION:
\"\"\"{question}\"\"\"

ANSWER:
\"\"\"{answer}\"\"\"
"""
