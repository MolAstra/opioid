from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opioid.baselines import config
from opioid.baselines.features import build_feature_dataset
from opioid.baselines.models import predict_probabilities

OUTPUT_DIR = Path("outputs/candidates")
FIGURES_DIR = OUTPUT_DIR / "figures"
RUNTIME_CONFIG_PATH = config.ARTIFACTS_DIR / "runtime_config.json"

DB_SOURCES = {
    "gpcrdb": {
        "path": Path("/home/silong/codex/tmp/tmp/datasets/screen_datasets/1_GPCRdb/gpcrdb.csv"),
        "smiles_col": "smiles",
    },
    "zinc": {
        "path": Path("/home/silong/codex/tmp/tmp/datasets/screen_datasets/2_ZINCdb/zinc_scaffold.csv"),
        "smiles_col": "smiles",
    },
    "reinvent": {
        "path": Path("/home/silong/codex/tmp/tmp/datasets/screen_datasets/3_reinvent/generation/stage2_1.csv"),
        "smiles_col": "SMILES",
    },
    "ouroboros": {
        "path": Path("/home/silong/codex/tmp/tmp/datasets/screen_datasets/4_ouroboros/opioid_ouroboros.csv"),
        "smiles_col": "smiles",
    },
}

OUTPUT_CSVS = {
    "gpcrdb": OUTPUT_DIR / "pred_gpcrdb.csv",
    "zinc": OUTPUT_DIR / "pred_zinc.csv",
    "reinvent": OUTPUT_DIR / "pred_reinvent.csv",
    "ouroboros": OUTPUT_DIR / "pred_ouroboros.csv",
}

STATS_CSV = OUTPUT_DIR / "candidate_stats.csv"
INVALID_SMILES_JSON = OUTPUT_DIR / "invalid_smiles.json"
RUN_CONFIG_JSON = OUTPUT_DIR / "run_config.json"

CONSENSUS_RATES_PNG = FIGURES_DIR / "candidate_consensus_rates.png"
COUNTS_PNG = FIGURES_DIR / "candidate_counts.png"
ECDF_PNG = FIGURES_DIR / "candidate_top3_mean_prob_ecdf.png"


def _load_training_smiles() -> set[str]:
    smiles_set: set[str] = set()
    for fold_idx in range(config.N_FOLDS):
        fold_dir = config.FOLD_DIR / f"fold_{fold_idx}"
        for split_name in ("train", "valid"):
            split_path = fold_dir / f"{split_name}.csv"
            if not split_path.exists():
                continue
            split_df = pd.read_csv(split_path, usecols=[config.SMILES_COLUMN])
            smiles_set.update(split_df[config.SMILES_COLUMN].dropna().astype(str))
    if config.INPUT_TEST_CSV.exists():
        test_df = pd.read_csv(config.INPUT_TEST_CSV, usecols=[config.SMILES_COLUMN])
        smiles_set.update(test_df[config.SMILES_COLUMN].dropna().astype(str))
    return smiles_set


class TopModelPredictor:
    def __init__(self, runtime_config_path: Path) -> None:
        self.runtime_path = Path(runtime_config_path)
        self.runtime = json.loads(self.runtime_path.read_text(encoding="utf-8"))
        self.top_models = list(self.runtime["top_models"])
        self.best_model = str(self.runtime["best_model"])
        self.threshold = float(self.runtime["threshold"])
        self.radius = int(self.runtime["feature"]["radius"])
        self.n_bits = int(self.runtime["feature"]["n_bits"])
        model_files = dict(self.runtime["model_files"])
        self.models = {
            name: joblib.load(Path(model_files[name]))
            for name in self.top_models
        }

    def _predict_probabilities_batched(self, model_name: str, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.zeros((0,), dtype=np.float64)

        batch_size = 512 if model_name == "tabpfn" else 8192
        chunks: list[np.ndarray] = []
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            chunk_probs = predict_probabilities(self.models[model_name], x[start:stop])
            chunks.append(np.asarray(chunk_probs, dtype=np.float64))
        return np.concatenate(chunks, axis=0)

    def predict_dataframe(self, source_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        work_df = source_df.copy()
        unique_df = work_df[[config.SMILES_COLUMN]].drop_duplicates().reset_index(drop=True)
        unique_df[config.LABEL_COLUMN] = 0

        dataset = build_feature_dataset(unique_df)
        prob_map = {
            name: self._predict_probabilities_batched(name, dataset.features)
            for name in self.top_models
        }

        unique_pred_df = dataset.rows.drop(columns=[config.LABEL_COLUMN], errors="ignore").reset_index(drop=True).copy()
        for name in self.top_models:
            probs = prob_map[name].astype(float)
            labels = (probs >= self.threshold).astype(int)
            unique_pred_df[f"{name}_prob"] = probs
            unique_pred_df[f"{name}_label"] = labels

        out_df = work_df.merge(unique_pred_df, on=config.SMILES_COLUMN, how="inner")

        top_prob_cols = [f"{name}_prob" for name in self.top_models]
        top_label_cols = [f"{name}_label" for name in self.top_models]
        out_df["top3_all1"] = out_df[top_label_cols].eq(1).all(axis=1).astype(int)
        out_df["top3_all0"] = out_df[top_label_cols].eq(0).all(axis=1).astype(int)
        out_df["top3_mean_prob"] = out_df[top_prob_cols].mean(axis=1)
        out_df["top3_min_prob"] = out_df[top_prob_cols].min(axis=1)
        out_df["top3_max_prob"] = out_df[top_prob_cols].max(axis=1)

        out_df = out_df.sort_values(
            by=["top3_mean_prob", "top3_min_prob", config.SMILES_COLUMN],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        out_df["top3_rank"] = np.arange(1, len(out_df) + 1, dtype=int)
        return out_df, dataset.invalid_smiles


def _load_source_csv(path: Path, smiles_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if smiles_col not in df.columns:
        raise ValueError(f"{path} is missing required column: {smiles_col}")
    if smiles_col != config.SMILES_COLUMN:
        df = df.rename(columns={smiles_col: config.SMILES_COLUMN})
    return df


def _filter_training_overlap(source_df: pd.DataFrame, training_smiles: set[str]) -> tuple[pd.DataFrame, int]:
    work_df = source_df.copy()
    work_df[config.SMILES_COLUMN] = work_df[config.SMILES_COLUMN].astype(str)
    overlap_mask = work_df[config.SMILES_COLUMN].isin(training_smiles)
    overlap_count = int(overlap_mask.sum())
    return work_df.loc[~overlap_mask].reset_index(drop=True), overlap_count


def _save_summary_plots(stats_df: pd.DataFrame, ecdf_frames: list[pd.DataFrame]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(stats_df))
    width = 0.36
    ax.bar(x - width / 2, stats_df["top3_all1_rate"], width=width, color="#c45d2f", edgecolor="none", label="Top3 all1")
    ax.bar(x + width / 2, stats_df["top3_all0_rate"], width=width, color="#4f7c53", edgecolor="none", label="Top3 all0")
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["db"])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CONSENSUS_RATES_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(stats_df))
    ax.bar(x, stats_df["n_source"], width=0.58, color="#b8c6d1", edgecolor="none", label="Source")
    ax.bar(x, stats_df["top3_all1_n"], width=0.38, color="#c45d2f", edgecolor="none", label="Top3 all1")
    ax.bar(x, stats_df["top3_all0_n"], width=0.22, color="#4f7c53", edgecolor="none", label="Top3 all0")
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["db"])
    ax.set_ylabel("Count")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(COUNTS_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for frame in ecdf_frames:
        values = frame["top3_mean_prob"].sort_values().reset_index(drop=True)
        if len(values) == 0:
            continue
        y = (values.index + 1) / len(values)
        ax.plot(values.to_numpy(dtype=float), y.to_numpy(dtype=float), linewidth=2.0, label=str(frame["db"].iloc[0]))
    ax.set_xlabel("Top3 mean probability")
    ax.set_ylabel("ECDF")
    ax.grid(color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ECDF_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _load_existing_stats() -> pd.DataFrame:
    if STATS_CSV.exists():
        return pd.read_csv(STATS_CSV)
    return pd.DataFrame(
        columns=[
            "db",
            "n_source",
            "n_overlap_filtered",
            "n_valid",
            "n_invalid",
            "top3_all1_n",
            "top3_all1_rate",
            "top3_all0_n",
            "top3_all0_rate",
        ]
    )


def _load_existing_invalid_smiles() -> dict[str, list[str]]:
    if INVALID_SMILES_JSON.exists():
        return json.loads(INVALID_SMILES_JSON.read_text(encoding="utf-8"))
    return {}


def _load_existing_ecdf_frames(selected_dbs: set[str]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for db_name, output_csv in OUTPUT_CSVS.items():
        if db_name in selected_dbs or not output_csv.exists():
            continue
        df = pd.read_csv(output_csv, usecols=["top3_mean_prob"])
        df["db"] = db_name
        frames.append(df)
    return frames


def run_candidate_library(selected_dbs: list[str] | None = None) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictor = TopModelPredictor(RUNTIME_CONFIG_PATH)
    training_smiles = _load_training_smiles()

    db_names = list(DB_SOURCES) if selected_dbs is None else selected_dbs
    selected_set = set(db_names)
    stats_df = _load_existing_stats()
    invalid_smiles = _load_existing_invalid_smiles()
    ecdf_frames: list[pd.DataFrame] = _load_existing_ecdf_frames(selected_set)

    for db_name in db_names:
        source_info = DB_SOURCES[db_name]
        source_path = Path(source_info["path"])
        source_df = _load_source_csv(source_path, str(source_info["smiles_col"]))
        filtered_df, overlap_count = _filter_training_overlap(source_df, training_smiles)
        pred_df, invalid = predictor.predict_dataframe(filtered_df)
        pred_df.to_csv(OUTPUT_CSVS[db_name], index=False)

        invalid_smiles[db_name] = invalid
        ecdf_frame = pred_df[["top3_mean_prob"]].copy()
        ecdf_frame["db"] = db_name
        ecdf_frames.append(ecdf_frame)

        n_source = int(len(source_df))
        n_valid = int(len(pred_df))
        n_invalid = int(len(invalid))
        top3_all1_n = int(pred_df["top3_all1"].sum())
        top3_all0_n = int(pred_df["top3_all0"].sum())
        stats_row = {
            "db": db_name,
            "n_source": n_source,
            "n_overlap_filtered": overlap_count,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "top3_all1_n": top3_all1_n,
            "top3_all1_rate": top3_all1_n / max(n_valid, 1),
            "top3_all0_n": top3_all0_n,
            "top3_all0_rate": top3_all0_n / max(n_valid, 1),
        }

        stats_df = stats_df[stats_df["db"] != db_name]
        stats_df = pd.concat([stats_df, pd.DataFrame([stats_row])], ignore_index=True)

    stats_df = stats_df.sort_values("db").reset_index(drop=True)
    stats_df.to_csv(STATS_CSV, index=False)
    INVALID_SMILES_JSON.write_text(json.dumps(invalid_smiles, indent=2), encoding="utf-8")
    RUN_CONFIG_JSON.write_text(
        json.dumps(
            {
                "sources": {
                    name: {
                        "path": str(source_info["path"]),
                        "smiles_col": str(source_info["smiles_col"]),
                    }
                    for name, source_info in DB_SOURCES.items()
                },
                "top_models": predictor.top_models,
                "best_model": predictor.best_model,
                "threshold": predictor.threshold,
                "feature": {
                    "type": "morgan",
                    "radius": predictor.radius,
                    "n_bits": predictor.n_bits,
                },
                "excluded_training_smiles": {
                    "n_unique": len(training_smiles),
                    "sources": [
                        str(config.FOLD_DIR / f"fold_{fold_idx}" / "train.csv")
                        for fold_idx in range(config.N_FOLDS)
                    ]
                    + [
                        str(config.FOLD_DIR / f"fold_{fold_idx}" / "valid.csv")
                        for fold_idx in range(config.N_FOLDS)
                    ]
                    + [str(config.INPUT_TEST_CSV)],
                },
                "output_dir": str(OUTPUT_DIR),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _save_summary_plots(stats_df, ecdf_frames)

    return {
        "output_dir": OUTPUT_DIR,
        "per_db_csvs": {db_name: OUTPUT_CSVS[db_name] for db_name in db_names},
        "stats_csv": STATS_CSV,
        "invalid_smiles_json": INVALID_SMILES_JSON,
        "run_config_json": RUN_CONFIG_JSON,
        "figures": {
            "consensus_rates": CONSENSUS_RATES_PNG,
            "counts": COUNTS_PNG,
            "ecdf": ECDF_PNG,
        },
    }
