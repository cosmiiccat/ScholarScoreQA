# led/config.py

import torch

class LEDConfig:
    MODEL_NAME = "allenai/led-base-16384"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # decoding
    MAX_GEN_LEN = 150

    # LED has max_length=16384 but working safe default
    MAX_INPUT_LEN = 4096
