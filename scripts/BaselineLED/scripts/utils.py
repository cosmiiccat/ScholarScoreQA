# led/utils.py

from transformers import LEDTokenizer, LEDForConditionalGeneration

def load_led(model_name, device):
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
    return tokenizer, model
