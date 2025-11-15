# config.py
from openai import OpenAI
import spacy

# Load spaCy model globally
SPACY_NLP = spacy.load("en_core_web_trf")

# OpenAI client
OPENAI_CLIENT = OpenAI(api_key="YOUR_API_KEY")
MODEL_NAME = "gpt-5"
