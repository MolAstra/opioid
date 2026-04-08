from __future__ import annotations

from pathlib import Path

INPUT_TEST_CSV = Path("datasets/cv/test.csv")
FOLD_DIR = Path("datasets/cv")
OUTPUT_DIR = Path("outputs/baselines")
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
TEST_PREDICTIONS_DIR = OUTPUT_DIR / "test_predictions"
CHEMPROP_WORK_DIR = OUTPUT_DIR / "chemprop"
STATUS_JSON = OUTPUT_DIR / "status.json"

SMILES_COLUMN = "smiles"
LABEL_COLUMN = "label"
SEED = 42
THRESHOLD = 0.5

N_FOLDS = 5
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

DEPLOYABLE_MODELS = ["rf", "svm", "xgb", "lgbm"]
BENCHMARK_MODELS = [
    "svm",
    "rf",
    "xgb",
    "lgbm",
    "tabpfn",
    "chemprop_default",
    "chemprop_morgan",
    "chemprop_morgan_rdkit2d",
]
