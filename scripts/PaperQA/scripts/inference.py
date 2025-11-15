# scripts/run_paperqa_inference.py

import sys
sys.path.append("..")

from .utils import set_openai_key
from .model import PaperQAModel

# ---- Set API Key ----
set_openai_key("YOUR_OPENAI_KEY_HERE")

# ---- Initialize Model ----
model = PaperQAModel()

# ---- Example Prompt ----
prompt_text = """<CONTEXT>"""

question = "<QUESTION>"

# ---- Run ----
answer = model.answer(prompt_text, question)
print("ANSWER:\n", answer)
