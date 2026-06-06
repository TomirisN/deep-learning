"""
Convert GloBI-style interaction TSV into natural-language texts for GPT-2.

The assignment references GBIF and EcoBase; this dataset comes from
Global Biotic Interactions (GloBI) — the interaction database cited in the
lab manual alongside GBIF. Rows are turned into English ecosystem sentences
because the base GPT-2 model is pretrained on English.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd

from config import (
    DATA_DIR,
    MAX_ROWS,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_TSV,
    TEST_TXT,
    TRAIN_RATIO,
    TRAIN_TXT,
    VAL_RATIO,
    VAL_TXT,
)

# Interaction types relevant to food webs / ecological relationships
VALID_INTERACTIONS = {
    "eats",
    "preysOn",
    "pollinates",
    "visitsFlowersOf",
    "interactsWith",
    "parasiteOf",
    "hasParasite",
    "symbiotOf",
    "mutualisticWith",
}

INTERACTION_PHRASES = {
    "eats": "feeds on",
    "preysOn": "preys on",
    "pollinates": "pollinates",
    "visitsFlowersOf": "visits the flowers of",
    "interactsWith": "interacts with",
    "parasiteOf": "parasitizes",
    "hasParasite": "is parasitized by",
    "symbiotOf": "lives in symbiosis with",
    "mutualisticWith": "forms a mutualistic relationship with",
}

# Rough habitat hints from taxonomy (EcoBase / GBIF-style context)
HABITAT_BY_CLASS = {
    "Mammalia": "In this terrestrial ecosystem,",
    "Aves": "In this avian ecosystem,",
    "Insecta": "In this insect-rich ecosystem,",
    "Teleostei": "In this marine food web,",
    "Actinopterygii": "In this aquatic ecosystem,",
    "Magnoliopsida": "In this plant community,",
    "Filicopsida": "In this forest understory,",
}

JUNK_PATTERN = re.compile(
    r"(same slab|BAITED|http://|guid/|^\s*leaves\s*$|^\s*Insecta\s*$|"
    r"^IP\.\d+|^MCZ:|^MSB:|^NPS:|^UAM:|^MVZ:|^CAS:|^USNM:)",
    re.IGNORECASE,
)


def is_valid_taxon(name: str | float) -> bool:
    if not isinstance(name, str):
        return False
    name = name.strip()
    if len(name) < 3:
        return False
    if JUNK_PATTERN.search(name):
        return False
    return True


def clean_class(value: str | float) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if value else None


def row_to_sentence(row: pd.Series) -> str | None:
    source = row["sourceTaxonName"]
    target = row["targetTaxonName"]
    itype = row["interactionTypeName"]

    if itype not in VALID_INTERACTIONS:
        return None
    if not is_valid_taxon(source) or not is_valid_taxon(target):
        return None

    phrase = INTERACTION_PHRASES[itype]
    src_class = clean_class(row.get("sourceTaxonClassName"))
    tgt_class = clean_class(row.get("targetTaxonClassName"))

    if src_class and tgt_class:
        return (
            f"In this ecosystem, {source} ({src_class}) {phrase} "
            f"{target} ({tgt_class})."
        )
    if src_class:
        return f"In this ecosystem, {source} ({src_class}) {phrase} {target}."
    return f"In this ecosystem, {source} {phrase} {target}."


def row_to_short(row: pd.Series) -> str | None:
    """Shorter variant — closer to assignment example style."""
    source = row["sourceTaxonName"]
    target = row["targetTaxonName"]
    itype = row["interactionTypeName"]

    if itype not in VALID_INTERACTIONS:
        return None
    if not is_valid_taxon(source) or not is_valid_taxon(target):
        return None

    phrase = INTERACTION_PHRASES[itype]
    src_class = clean_class(row.get("sourceTaxonClassName"))
    prefix = HABITAT_BY_CLASS.get(src_class, "In this ecosystem,") if src_class else "In this ecosystem,"
    return f"{prefix} {source} {phrase} {target}."


def build_food_web_paragraph(group: pd.DataFrame) -> str | None:
    """EcoBase-style mini food-web description from a species group."""
    sentences: list[str] = []
    habitat = "In this ecosystem,"

    for _, row in group.iterrows():
        src_class = clean_class(row.get("sourceTaxonClassName"))
        if src_class and src_class in HABITAT_BY_CLASS:
            habitat = HABITAT_BY_CLASS[src_class]
            break

    for _, row in group.head(4).iterrows():
        s = row_to_sentence(row)
        if s:
            sentences.append(s.replace("In this ecosystem, ", ""))

    if len(sentences) < 2:
        return None

    body = " ".join(sentences)
    return f"{habitat} the food web includes: {body}"


def load_and_filter(path: Path, max_rows: int | None) -> pd.DataFrame:
    print(f"Reading {path} ...")
    df = pd.read_csv(path, sep="\t", nrows=max_rows, low_memory=False)
    print(f"Loaded {len(df):,} rows")

    df = df.dropna(subset=["sourceTaxonName", "targetTaxonName", "interactionTypeName"])
    df = df[df["interactionTypeName"].isin(VALID_INTERACTIONS)]
    df = df[
        df["sourceTaxonName"].apply(is_valid_taxon)
        & df["targetTaxonName"].apply(is_valid_taxon)
    ]
    print(f"After filtering: {len(df):,} rows")
    return df


def build_corpus(df: pd.DataFrame) -> list[str]:
    texts: list[str] = []

    for _, row in df.iterrows():
        for fn in (row_to_sentence, row_to_short):
            text = fn(row)
            if text:
                texts.append(text)

    # EcoBase-style grouped paragraphs by source genus
    grouped = df.groupby("sourceTaxonGenusName", dropna=True)
    for _, group in grouped:
        if len(group) >= 2:
            paragraph = build_food_web_paragraph(group)
            if paragraph:
                texts.append(paragraph)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    print(f"Built {len(unique):,} unique training texts")
    return unique


def split_and_save(texts: list[str], out_dir: Path) -> None:
    random.seed(RANDOM_SEED)
    random.shuffle(texts)

    n = len(texts)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train.txt": texts[:n_train],
        "val.txt": texts[n_train : n_train + n_val],
        "test.txt": texts[n_train + n_val :],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, chunk in splits.items():
        path = out_dir / filename
        path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        print(f"  {filename}: {len(chunk):,} lines -> {path}")


def main(max_rows: int | None = MAX_ROWS) -> None:
    if not RAW_TSV.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_TSV}")

    df = load_and_filter(RAW_TSV, max_rows)
    texts = build_corpus(df)
    if len(texts) < 100:
        raise RuntimeError("Too few texts after preprocessing. Check the TSV file.")

    split_and_save(texts, DATA_DIR)
    print("\nDone. Next step: python src/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ecosystem text corpus")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=MAX_ROWS,
        help="Max TSV rows to read (default: all configured in config.py)",
    )
    args = parser.parse_args()
    main(max_rows=args.max_rows)
