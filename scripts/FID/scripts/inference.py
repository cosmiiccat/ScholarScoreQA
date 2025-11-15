# scripts/run_fid_inference.py

import sys
sys.path.append("..")

from fid.config import FiDConfig
from fid.utils import load_fid_model
from fid.model import FiDModel

# Load settings
config = FiDConfig()

# Load model + tokenizer
tokenizer, model = load_fid_model(config.MODEL_NAME, config.DEVICE)

# Initialize FiD
fid = FiDModel(tokenizer, model, config)

# Example QA input
question = "<QUESTION>"
passage = """<CONTEXT>"""

# Run inference
answer = fid.answer(question, passage)
print("ANSWER:", answer)
