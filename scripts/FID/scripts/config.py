# fid/config.py

import torch

class FiDConfig:
    MODEL_NAME = "t5-base"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # chunking
    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 32
    MAX_PASSAGES = 6

    # decoding
    MAX_ANSWER_LEN = 64
    NUM_BEAMS = 4