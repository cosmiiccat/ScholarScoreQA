# scripts/run_led_inference.py

import sys
sys.path.append("..")

from led.config import LEDConfig
from led.utils import load_led
from led.model import LEDModel

# Load Config
config = LEDConfig()

# Load model + tokenizer
tokenizer, model = load_led(config.MODEL_NAME, config.DEVICE)

# Initialize LED Wrapper
led = LEDModel(tokenizer, model, config)

# ---- Test Example ----
input_text = """
<QUESTION>

<CONTEXT>
"""

answer = led.answer(input_text)
print("ANSWER:", answer)
