"""Fine-tune GPT-2 on ecosystem description texts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

from config import (
    BASE_MODEL_NAME,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    MAX_TRAIN_SAMPLES,
    MODEL_DIR,
    NUM_TRAIN_EPOCHS,
    RESULTS_DIR,
    TRAIN_TXT,
    VAL_TXT,
)


def tokenize_dataset(
    tokenizer: GPT2Tokenizer,
    train_path: Path,
    val_path: Path,
    max_samples: int | None = None,
):
    data_files = {"train": str(train_path)}
    if val_path.exists():
        data_files["validation"] = str(val_path)

    dataset = load_dataset("text", data_files=data_files)

    if max_samples and max_samples < len(dataset["train"]):
        dataset["train"] = dataset["train"].select(range(max_samples))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return tokenized


def main(
    num_epochs: int = NUM_TRAIN_EPOCHS,
    output_dir: Path = MODEL_DIR,
    max_samples: int | None = MAX_TRAIN_SAMPLES,
) -> None:
    if not TRAIN_TXT.exists():
        print(f"Missing {TRAIN_TXT}. Run: python src/prepare_data.py")
        sys.exit(1)

    print(f"Loading tokenizer and model: {BASE_MODEL_NAME}")
    tokenizer = GPT2Tokenizer.from_pretrained(BASE_MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenize_dataset(tokenizer, TRAIN_TXT, VAL_TXT, max_samples=max_samples)
    print(f"Train samples: {len(tokenized['train']):,}")
    if "validation" in tokenized:
        print(f"Validation samples: {len(tokenized['validation']):,}")

    use_cuda = torch.cuda.is_available()
    print(f"Device: {'cuda' if use_cuda else 'cpu'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps" if "validation" in tokenized else "no",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end="validation" in tokenized,
        report_to="none",
        fp16=use_cuda,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=data_collator,
    )

    print("\nStarting fine-tuning ...")
    train_result = trainer.train()

    print(f"\nSaving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime_sec": train_result.metrics.get("train_runtime"),
        "epochs": num_epochs,
        "train_samples": len(tokenized["train"]),
        "device": "cuda" if use_cuda else "cpu",
    }

    if "validation" in tokenized:
        eval_metrics = trainer.evaluate()
        metrics["eval_loss"] = eval_metrics["eval_loss"]
        metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])

    metrics_path = RESULTS_DIR / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to {metrics_path}")
    print("Next step: python src/generate.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on ecosystem texts")
    parser.add_argument("--epochs", type=int, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=MAX_TRAIN_SAMPLES,
        help="Limit training samples (use e.g. 2000 for a quick test)",
    )
    args = parser.parse_args()
    main(num_epochs=args.epochs, output_dir=args.output_dir, max_samples=args.max_samples)
