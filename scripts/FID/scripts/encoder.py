# fid/encoder.py

import torch
from transformers.modeling_outputs import BaseModelOutput

class FiDEncoder:
    """
    Encodes multiple chunks independently for FiD fusion.
    """

    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @torch.no_grad()
    def encode_chunks(self, question, chunks):
        encoded_states = []

        for chunk in chunks:
            text = f"{question} </s> {chunk}"

            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="longest"
            ).to(self.device)

            out = self.model.get_encoder()(**enc)
            hidden = out[0]  # tuple → take last_hidden_state

            encoded_states.append(hidden)

        # FiD fusion → concat hidden states along sequence dimension
        fused = torch.cat(encoded_states, dim=1)
        return BaseModelOutput(last_hidden_state=fused)
