"""Shared paths and hyperparameters for Task 4."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_TSV = PROJECT_ROOT / "interactions_sample_200k.tsv" / "interactions_sample_200k.tsv"

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_TXT = DATA_DIR / "train.txt"
VAL_TXT = DATA_DIR / "val.txt"
TEST_TXT = DATA_DIR / "test.txt"

MODEL_DIR = PROJECT_ROOT / "models" / "gpt2-ecosystem"
BASE_MODEL_NAME = "gpt2"

RESULTS_DIR = PROJECT_ROOT / "results"
GENERATED_FILE = RESULTS_DIR / "generated_descriptions.txt"
METRICS_FILE = RESULTS_DIR / "metrics.json"
MOS_TEMPLATE = RESULTS_DIR / "mos_evaluation_template.csv"

# Data preparation
MAX_ROWS = 50_000          # cap raw rows read from TSV (full file is 200k)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42

# Training (tuned for CPU; increase on GPU)
MAX_SEQ_LENGTH = 128
NUM_TRAIN_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
MAX_TRAIN_SAMPLES = None   # None = use all prepared texts

# Generation
GENERATION_PROMPTS = [
    "In this forest ecosystem,",
    "In this marine food web,",
    "In this ecosystem, predators",
    "The food chain includes",
    "Bees and flowers interact as",
    "In a freshwater ecosystem,",
]

GENERATION_KWARGS = {
    "max_new_tokens": 80,
    "num_return_sequences": 3,
    "temperature": 0.85,
    "top_p": 0.92,
    "do_sample": True,
    "pad_token_id": 50256,
}
