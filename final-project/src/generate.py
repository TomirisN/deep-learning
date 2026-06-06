"""Generate new ecosystem descriptions with fine-tuned GPT-2."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

from config import (
    BASE_MODEL_NAME,
    GENERATED_FILE,
    GENERATION_KWARGS,
    GENERATION_PROMPTS,
    MODEL_DIR,
    RESULTS_DIR,
)


def load_generator(model_dir: Path):
    if model_dir.exists() and (model_dir / "config.json").exists():
        print(f"Loading fine-tuned model from {model_dir}")
        tokenizer = GPT2Tokenizer.from_pretrained(str(model_dir))
        model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    else:
        print(f"Fine-tuned model not found at {model_dir}")
        print(f"Using base model '{BASE_MODEL_NAME}' (run train.py first for better results)")
        tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_NAME)
        model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def generate_all(generator, prompts: list[str]) -> list[dict]:
    results = []
    for prompt in prompts:
        outputs = generator(prompt, **GENERATION_KWARGS)
        for i, out in enumerate(outputs):
            results.append(
                {
                    "prompt": prompt,
                    "variant": i + 1,
                    "text": out["generated_text"],
                }
            )
    return results


def main(model_dir: Path = MODEL_DIR, output_file: Path = GENERATED_FILE) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generator = load_generator(model_dir)
    results = generate_all(generator, GENERATION_PROMPTS)

    lines = [
        f"# Generated ecosystem descriptions",
        f"# Model: {model_dir}",
        f"# Date: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for r in results:
        lines.append(f"Prompt: {r['prompt']}")
        lines.append(f"Generated ({r['variant']}):")
        lines.append(r["text"])
        lines.append("-" * 60)
        lines.append("")

    output_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(results)} generations to {output_file}")
    print("\nSample outputs:\n")
    for r in results[:3]:
        print(r["text"])
        print("-" * 40)

    print("\nNext step: python src/evaluate.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ecosystem descriptions")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=GENERATED_FILE)
    args = parser.parse_args()
    main(model_dir=args.model_dir, output_file=args.output)
