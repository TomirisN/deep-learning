"""
Minimal demo from the lab manual — run before or after training.

Before training: uses base GPT-2 (generic text).
After training:  uses fine-tuned model from models/gpt2-ecosystem/.
"""

from pathlib import Path
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models" / "gpt2-ecosystem"


def load_model():
    if MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists():
        print(f"Using fine-tuned model: {MODEL_DIR}")
        name = str(MODEL_DIR)
    else:
        print("Fine-tuned model not found — using base gpt2")
        print("Run: python src/train.py")
        name = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(name)
    model = GPT2LMHeadModel.from_pretrained(name)
    return tokenizer, model


def generate(prompt: str, max_new_tokens: int = 60) -> str:
    tokenizer, model = load_model()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "In this ecosystem, predators"
    print(f"Prompt: {prompt}\n")
    print(generate(prompt))
