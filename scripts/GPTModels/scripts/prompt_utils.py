# utils/prompt_utils.py

def build_prompt(context: str, question: str):
    """
    Build a clean, research-friendly prompt for QA.
    """

    prompt = f"""<PROMPT>"

Context:
\"\"\"{context}\"\"\"

Question:
{question}

Answer in one short, direct sentence.
"""
    return prompt.strip()
