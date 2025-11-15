# fid/model.py

import torch
from fid.chunker import PassageChunker
from fid.encoder import FiDEncoder

class FiDModel:
    """
    Research-grade Fusion-in-Decoder implementation using T5.
    """

    def __init__(self, tokenizer, model, config):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config

        # utilities
        self.chunker = PassageChunker(
            tokenizer,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
        self.encoder = FiDEncoder(
            tokenizer, model, device=config.DEVICE
        )

    def answer(self, question, passage):
        chunks = self.chunker.chunk(passage)
        chunks = chunks[: self.config.MAX_PASSAGES]

        if len(chunks) == 0:
            chunks = [""]

        encoder_outputs = self.encoder.encode_chunks(question, chunks)

        output_ids = self.model.generate(
            encoder_outputs=encoder_outputs,
            max_length=self.config.MAX_ANSWER_LEN,
            num_beams=self.config.NUM_BEAMS,
            early_stopping=True,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
