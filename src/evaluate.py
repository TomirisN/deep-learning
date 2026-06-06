"""
Evaluate text quality: Perplexity on test set + MOS evaluation template.

Perplexity is computed automatically.
MOS (Mean Opinion Score) requires human ratings — this script creates a CSV
template and computes MOS once you fill in the scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config import (
    BASE_MODEL_NAME,
    GENERATED_FILE,
    MAX_SEQ_LENGTH,
    METRICS_FILE,
    MODEL_DIR,
    MOS_TEMPLATE,
    RESULTS_DIR,
    TEST_TXT,
)


def compute_perplexity(model_dir: Path, test_path: Path) -> dict:
    if not test_path.exists():
        raise FileNotFoundError(f"Test file missing: {test_path}. Run prepare_data.py first.")

    tokenizer = GPT2Tokenizer.from_pretrained(str(model_dir))
    model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = load_dataset("text", data_files={"test": str(test_path)})["test"]
    texts = dataset["text"]

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )
            input_ids = encodings.input_ids.to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            n_tokens = input_ids.size(1)
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)

    return {
        "model_dir": str(model_dir),
        "test_samples": len(texts),
        "avg_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 2),
    }


def create_mos_template(generated_file: Path, output_csv: Path) -> None:
    if not generated_file.exists():
        print(f"No generations at {generated_file}. Run generate.py first.")
        return

    texts: list[str] = []
    current: list[str] = []

    for line in generated_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("Generated"):
            current = []
        elif line.strip() == "-" * 60:
            if current:
                texts.append(" ".join(current).strip())
                current = []
        elif line.startswith("Prompt:") or line.startswith("#") or not line.strip():
            continue
        else:
            current.append(line.strip())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "generated_text",
                "coherence_1_to_5",
                "ecological_realism_1_to_5",
                "overall_1_to_5",
                "notes",
            ]
        )
        for i, text in enumerate(texts, start=1):
            writer.writerow([i, text, "", "", "", ""])

    print(f"MOS template saved: {output_csv}")
    print("Fill scores 1-5, then run: python src/evaluate.py --compute-mos")


def compute_mos_from_csv(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None

    scores: list[float] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in ("coherence_1_to_5", "ecological_realism_1_to_5", "overall_1_to_5"):
                val = row.get(col, "").strip()
                if val:
                    try:
                        scores.append(float(val))
                    except ValueError:
                        pass

    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def compare_base_vs_finetuned(test_path: Path) -> dict:
    """Compare perplexity: base GPT-2 vs fine-tuned model."""
    comparison = {}
    if MODEL_DIR.exists():
        comparison["fine_tuned"] = compute_perplexity(MODEL_DIR, test_path)

    print("Computing base model perplexity (may download ~500 MB on first run) ...")
    base_metrics = compute_perplexity(Path(BASE_MODEL_NAME), test_path)
    comparison["base_gpt2"] = base_metrics
    return comparison


def main(compute_mos: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_TXT.exists():
        print(f"Missing {TEST_TXT}. Run prepare_data.py first.")
        sys.exit(1)

    if not MODEL_DIR.exists():
        print(f"Missing fine-tuned model at {MODEL_DIR}. Run train.py first.")
        sys.exit(1)

    print("=== Perplexity evaluation ===")
    comparison = compare_base_vs_finetuned(TEST_TXT)

    print("\n--- Base GPT-2 ---")
    print(json.dumps(comparison["base_gpt2"], indent=2))
    if "fine_tuned" in comparison:
        print("\n--- Fine-tuned model ---")
        print(json.dumps(comparison["fine_tuned"], indent=2))
        base_ppl = comparison["base_gpt2"]["perplexity"]
        ft_ppl = comparison["fine_tuned"]["perplexity"]
        improvement = round((base_ppl - ft_ppl) / base_ppl * 100, 1)
        print(f"\nPerplexity improved by {improvement}% (lower is better)")

    create_mos_template(GENERATED_FILE, MOS_TEMPLATE)

    metrics = {"perplexity_comparison": comparison}

    if compute_mos:
        mos = compute_mos_from_csv(MOS_TEMPLATE)
        if mos is not None:
            metrics["mean_opinion_score"] = mos
            print(f"\nMOS (Mean Opinion Score): {mos}")
        else:
            print("\nMOS: fill in mos_evaluation_template.csv first")

    training_metrics = RESULTS_DIR / "training_metrics.json"
    if training_metrics.exists():
        metrics["training"] = json.loads(training_metrics.read_text(encoding="utf-8"))

    METRICS_FILE.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAll metrics saved to {METRICS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate perplexity and MOS")
    parser.add_argument(
        "--compute-mos",
        action="store_true",
        help="Compute MOS after filling mos_evaluation_template.csv",
    )
    args = parser.parse_args()
    main(compute_mos=args.compute_mos)
