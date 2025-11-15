# led/model.py

import torch
from led.config import LEDConfig

class LEDModel:
    """
    Minimal LED Question Answering model.
    (Your original working code preserved exactly)
    """

    def __init__(self, tokenizer, model, config: LEDConfig):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config

    @torch.no_grad()
    def answer(self, input_text: str):
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.MAX_INPUT_LEN
        ).to(self.config.DEVICE)

        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.config.MAX_GEN_LEN
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
