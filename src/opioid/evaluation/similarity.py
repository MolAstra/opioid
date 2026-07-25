"""Post-hoc similarity diagnostics for persisted DPBench local partitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SMILES_COLUMN = "smiles"
LABEL_COLUMN = "label"
SAMPLE_ID_COLUMN = "sample_id"
N_FOLDS = 5
MORGAN_RADIUS = 2
MORGAN_BITS = 1024
REPORT_THRESHOLD = 0.70
HISTOGRAM_BINS = 100


@dataclass(frozen=True)
class PairSimilarityDiagnostics:
    """Exact similarities from target rows to all source rows in one comparison."""

    comparison: str
    per_target: pd.DataFrame
    pairwise_similarities: np.ndarray
    summary: dict[str, int | float]


@dataclass(frozen=True)
class FoldSimilarityDiagnostics:
    """Train/validation similarity diagnostics for one CV fold."""

    fold: int
    per_validation: pd.DataFrame
    pairwise_similarities: np.ndarray
    summary: dict[str, int | float]


@dataclass(frozen=True)
class CrossValidationSimilarityDiagnostics:
    """Supplemental train/validation diagnostics for all persisted CV folds."""

    protocol: str
    split_dir: Path
    threshold: float
    folds: tuple[FoldSimilarityDiagnostics, ...]

    @property
    def per_validation(self) -> pd.DataFrame:
        return pd.concat([fold.per_validation for fold in self.folds], ignore_index=True)

    @property
    def fold_summary(self) -> pd.DataFrame:
        return pd.DataFrame([fold.summary for fold in self.folds])


@dataclass(frozen=True)
class ProtocolSimilarityDiagnostics:
    """Fixed-test and five-fold similarity diagnostics for one DPBench protocol."""

    protocol: str
    split_dir: Path
    threshold: float
    test: PairSimilarityDiagnostics
    folds: tuple[FoldSimilarityDiagnostics, ...]
    datasail_diagnostics: dict[str, Any] | None
    nonempty_scaffold_overlap: int

    @property
    def per_validation(self) -> pd.DataFrame:
        return pd.concat([fold.per_validation for fold in self.folds], ignore_index=True)

    @property
    def fold_summary(self) -> pd.DataFrame:
        return pd.DataFrame([fold.summary for fold in self.folds])


def diagnose_protocol_split(
    split_dir: Path | str,
    *,
    protocol: str | None = None,
) -> ProtocolSimilarityDiagnostics:
    """Audit a persisted 5CV + shared-test split without imposing a hard cutoff.

    DataSAIL optimizes the fixed-test/CV allocation, but it does not promise a
    hard Tanimoto threshold for any reported pair. The threshold is therefore
    descriptive only, never a leakage assertion.
    """

    resolved_dir = Path(split_dir)
    resolved_protocol = protocol or resolved_dir.name.removeprefix("processed_")
    test = _load_frame(resolved_dir / "test.csv")
    folds, development = _diagnose_folds(resolved_dir)

    development_frame = _unique_development(development)
    test_diagnostics = _diagnose_pair(
        source=development_frame,
        target=test,
        comparison="development_to_test",
        threshold=REPORT_THRESHOLD,
    )
    diagnostics = _load_json(resolved_dir / "diagnostics.json")
    overlap = _nonempty_scaffold_overlap(development_frame, test)
    result = ProtocolSimilarityDiagnostics(
        protocol=resolved_protocol,
        split_dir=resolved_dir,
        threshold=REPORT_THRESHOLD,
        test=test_diagnostics,
        folds=tuple(folds),
        datasail_diagnostics=diagnostics if resolved_protocol == "datasail" else None,
        nonempty_scaffold_overlap=overlap,
    )
    if resolved_protocol == "scaffold":
        assert_scaffold_test_isolation(result)
    return result


def diagnose_cv_folds(
    split_dir: Path | str, *, protocol: str | None = None
) -> CrossValidationSimilarityDiagnostics:
    """Audit only train/validation folds without reading the fixed test table."""

    resolved_dir = Path(split_dir)
    resolved_protocol = protocol or resolved_dir.name.removeprefix("processed_")
    folds, _ = _diagnose_folds(resolved_dir)
    return CrossValidationSimilarityDiagnostics(
        protocol=resolved_protocol,
        split_dir=resolved_dir,
        threshold=REPORT_THRESHOLD,
        folds=tuple(folds),
    )


def assert_scaffold_test_isolation(result: ProtocolSimilarityDiagnostics) -> None:
    """Ensure a scaffold partition has no non-empty development/test overlap."""

    if result.nonempty_scaffold_overlap:
        raise AssertionError(
            "Scaffold development/test isolation failed: "
            f"{result.nonempty_scaffold_overlap} non-empty scaffold(s) occur in development and test."
        )


def write_similarity_diagnostics(
    result: ProtocolSimilarityDiagnostics,
    output_dir: Path | str,
    *,
    histogram_bins: int = HISTOGRAM_BINS,
) -> dict[str, Path]:
    """Write exact CSV/JSON tables and figures for reviewer-facing diagnostics."""

    if histogram_bins < 2:
        raise ValueError("histogram_bins must be at least 2")
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    validation_path = output_dir / "validation_to_train_similarity.csv"
    fold_summary_path = output_dir / "fold_similarity_summary.csv"
    test_path = output_dir / "test_to_development_similarity.csv"
    test_summary_path = output_dir / "test_similarity_summary.csv"
    histogram_path = output_dir / "train_valid_pair_similarity_histogram.csv"
    summary_path = output_dir / "similarity_summary.json"
    max_figure = figures_dir / "train_valid_max_similarity.png"
    test_figure = figures_dir / "test_development_max_similarity.png"
    ecdf_figure = figures_dir / "train_valid_pair_similarity_ecdf.png"

    result.per_validation.to_csv(validation_path, index=False)
    result.fold_summary.to_csv(fold_summary_path, index=False)
    result.test.per_target.to_csv(test_path, index=False)
    pd.DataFrame([result.test.summary]).to_csv(test_summary_path, index=False)
    _pairwise_histogram(result, histogram_bins).to_csv(histogram_path, index=False)
    summary_path.write_text(
        json.dumps(_protocol_summary_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_fold_maxima(result, max_figure)
    _plot_test_maxima(result, test_figure)
    _plot_fold_ecdf(result, ecdf_figure)
    return {
        "per_validation_csv": validation_path,
        "fold_summary_csv": fold_summary_path,
        "test_similarity_csv": test_path,
        "test_summary_csv": test_summary_path,
        "histogram_csv": histogram_path,
        "summary_json": summary_path,
        "max_similarity_figure": max_figure,
        "test_max_similarity_figure": test_figure,
        "pair_ecdf_figure": ecdf_figure,
    }


def write_cv_fold_similarity_diagnostics(
    result: CrossValidationSimilarityDiagnostics,
    output_dir: Path | str,
    *,
    histogram_bins: int = HISTOGRAM_BINS,
) -> dict[str, Path]:
    """Write only supplemental train/validation evidence for a fixed-test CV partition."""

    if histogram_bins < 2:
        raise ValueError("histogram_bins must be at least 2")
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "validation_to_train_similarity.csv"
    fold_summary_path = output_dir / "fold_similarity_summary.csv"
    histogram_path = output_dir / "train_valid_pair_similarity_histogram.csv"
    summary_path = output_dir / "similarity_summary.json"
    max_figure = figures_dir / "train_valid_max_similarity.png"
    ecdf_figure = figures_dir / "train_valid_pair_similarity_ecdf.png"

    result.per_validation.to_csv(validation_path, index=False)
    result.fold_summary.to_csv(fold_summary_path, index=False)
    _pairwise_histogram(result, histogram_bins).to_csv(histogram_path, index=False)
    summary_path.write_text(
        json.dumps(_cv_summary_payload(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_fold_maxima(result, max_figure)
    _plot_fold_ecdf(result, ecdf_figure)
    return {
        "per_validation_csv": validation_path,
        "fold_summary_csv": fold_summary_path,
        "histogram_csv": histogram_path,
        "summary_json": summary_path,
        "max_similarity_figure": max_figure,
        "pair_ecdf_figure": ecdf_figure,
    }


def _diagnose_folds(
    split_dir: Path,
) -> tuple[list[FoldSimilarityDiagnostics], list[pd.DataFrame]]:
    folds: list[FoldSimilarityDiagnostics] = []
    development: list[pd.DataFrame] = []
    for fold_index in range(N_FOLDS):
        train_path, valid_path = _fold_paths(split_dir, fold_index)
        train = _load_frame(train_path)
        valid = _load_frame(valid_path)
        development.extend((train, valid))
        comparison = _diagnose_pair(
            source=train,
            target=valid,
            comparison=f"fold_{fold_index}_train_to_valid",
            threshold=REPORT_THRESHOLD,
        )
        folds.append(
            FoldSimilarityDiagnostics(
                fold=fold_index,
                per_validation=comparison.per_target.assign(fold=fold_index),
                pairwise_similarities=comparison.pairwise_similarities,
                summary={"fold": fold_index, **comparison.summary},
            )
        )
    return folds, development


def _fold_paths(split_dir: Path, fold_index: int) -> tuple[Path, Path]:
    dpbench = (
        split_dir / f"train_fold-{fold_index}.csv",
        split_dir / f"valid_fold-{fold_index}.csv",
    )
    if dpbench[0].exists() or dpbench[1].exists():
        return dpbench
    legacy_dir = split_dir / f"fold_{fold_index}"
    return legacy_dir / "train.csv", legacy_dir / "valid.csv"


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required split file does not exist: {path}")
    frame = pd.read_csv(path)
    missing = {SMILES_COLUMN, LABEL_COLUMN}.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty or frame[SMILES_COLUMN].isna().any():
        raise ValueError(f"{path} contains empty or missing SMILES rows")
    return frame


def _unique_development(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    subset = (
        [SAMPLE_ID_COLUMN]
        if SAMPLE_ID_COLUMN in combined.columns
        else [SMILES_COLUMN, LABEL_COLUMN]
    )
    return combined.drop_duplicates(subset=subset).reset_index(drop=True)


def _diagnose_pair(
    *,
    source: pd.DataFrame,
    target: pd.DataFrame,
    comparison: str,
    threshold: float,
) -> PairSimilarityDiagnostics:
    source_fingerprints = _morgan_fingerprints(source[SMILES_COLUMN].tolist(), comparison)
    target_fingerprints = _morgan_fingerprints(target[SMILES_COLUMN].tolist(), comparison)
    records: list[dict[str, int | float | str]] = []
    pairwise: list[np.ndarray] = []
    for index, fingerprint in enumerate(target_fingerprints):
        values = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, source_fingerprints), dtype=float
        )
        nearest = int(np.argmax(values))
        row = target.iloc[index]
        record: dict[str, int | float | str] = {
            "comparison": comparison,
            "target_row_index": index,
            "target_smiles": str(row[SMILES_COLUMN]),
            "target_label": int(row[LABEL_COLUMN]),
            "n_source": len(source),
            "nearest_source_row_index": nearest,
            "nearest_source_smiles": str(source.iloc[nearest][SMILES_COLUMN]),
            "max_source_tanimoto": float(np.max(values)),
            "mean_source_tanimoto": float(np.mean(values)),
            "n_source_at_or_above_report_threshold": int(np.count_nonzero(values >= threshold)),
        }
        if SAMPLE_ID_COLUMN in target.columns:
            record["target_sample_id"] = str(row[SAMPLE_ID_COLUMN])
        records.append(record)
        pairwise.append(values)
    similarities = np.concatenate(pairwise)
    per_target = pd.DataFrame.from_records(records)
    maxima = per_target["max_source_tanimoto"].to_numpy(dtype=float)
    summary = {
        "comparison": comparison,
        "n_source": int(len(source)),
        "n_target": int(len(target)),
        "n_cross_pairs": int(len(similarities)),
        "pair_tanimoto_mean": float(np.mean(similarities)),
        "pair_tanimoto_median": float(np.median(similarities)),
        "pair_tanimoto_std": float(np.std(similarities)),
        "pair_tanimoto_min": float(np.min(similarities)),
        "pair_tanimoto_max": float(np.max(similarities)),
        "target_max_tanimoto_mean": float(np.mean(maxima)),
        "target_max_tanimoto_median": float(np.median(maxima)),
        "target_max_tanimoto_min": float(np.min(maxima)),
        "target_max_tanimoto_max": float(np.max(maxima)),
        "n_cross_pairs_at_or_above_report_threshold": int(
            np.count_nonzero(similarities >= threshold)
        ),
        "n_targets_with_source_at_or_above_report_threshold": int(
            np.count_nonzero(maxima >= threshold)
        ),
    }
    return PairSimilarityDiagnostics(comparison, per_target, similarities, summary)


def _morgan_fingerprints(smiles_values: list[str], comparison: str) -> list[Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    fingerprints: list[Any] = []
    invalid: list[int] = []
    for row_index, smiles in enumerate(smiles_values):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            invalid.append(row_index)
        else:
            fingerprints.append(generator.GetFingerprint(molecule))
    if invalid:
        raise ValueError(f"Invalid SMILES in {comparison} at rows {invalid[:10]}")
    return fingerprints


def _nonempty_scaffold_overlap(development: pd.DataFrame, test: pd.DataFrame) -> int:
    def scaffolds(frame: pd.DataFrame) -> set[str]:
        if "scaffold" in frame.columns:
            return {value for value in frame["scaffold"].fillna("").astype(str) if value}
        return {
            MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
            for molecule in (Chem.MolFromSmiles(smiles) for smiles in frame[SMILES_COLUMN])
            if molecule is not None
            and MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        }

    return len(scaffolds(development).intersection(scaffolds(test)))


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _pairwise_histogram(
    result: ProtocolSimilarityDiagnostics | CrossValidationSimilarityDiagnostics,
    histogram_bins: int,
) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, histogram_bins + 1)
    rows: list[dict[str, int | float]] = []
    for fold in result.folds:
        counts, _ = np.histogram(fold.pairwise_similarities, bins=edges)
        total = len(fold.pairwise_similarities)
        rows.extend(
            {
                "fold": fold.fold,
                "bin_left": float(left),
                "bin_right": float(right),
                "n_pairs": int(count),
                "fraction_pairs": float(count / total),
            }
            for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True)
        )
    return pd.DataFrame.from_records(rows)


def _protocol_summary_payload(result: ProtocolSimilarityDiagnostics) -> dict[str, Any]:
    fold_summary = result.fold_summary
    return {
        "protocol": result.protocol,
        "split_dir": str(result.split_dir),
        "fingerprint": {"type": "Morgan", "radius": MORGAN_RADIUS, "n_bits": MORGAN_BITS},
        "report_threshold": result.threshold,
        "n_folds": len(result.folds),
        "test": result.test.summary,
        "cv": {
            "n_validation_molecule_evaluations": int(len(result.per_validation)),
            "n_train_valid_cross_pairs": int(fold_summary["n_cross_pairs"].sum()),
            "max_cross_pair_tanimoto": float(fold_summary["pair_tanimoto_max"].max()),
        },
        "nonempty_scaffold_overlap": result.nonempty_scaffold_overlap,
        "scaffold_test_assertion": "passed" if result.protocol == "scaffold" else "not_applicable",
        "datasail_diagnostics": result.datasail_diagnostics,
        "interpretation": (
            "Similarity values are descriptive post-hoc diagnostics. DataSAIL does not impose "
            "a hard Morgan threshold on inner train/validation folds."
        ),
    }


def _cv_summary_payload(result: CrossValidationSimilarityDiagnostics) -> dict[str, Any]:
    fold_summary = result.fold_summary
    return {
        "schema_version": 1,
        "kind": "muorscreen.cv_fold_similarity_supplement",
        "scope": "train_validation_folds_only",
        "protocol": result.protocol,
        "split_dir": str(result.split_dir),
        "fingerprint": {"type": "Morgan", "radius": MORGAN_RADIUS, "n_bits": MORGAN_BITS},
        "report_threshold": result.threshold,
        "n_folds": len(result.folds),
        "n_validation_molecule_evaluations": int(len(result.per_validation)),
        "n_train_valid_cross_pairs": int(fold_summary["n_cross_pairs"].sum()),
        "max_cross_pair_tanimoto": float(fold_summary["pair_tanimoto_max"].max()),
        "canonical_global_evidence": "../dpnet_analysis/report.json",
        "interpretation": (
            "Supplemental per-fold evidence only. DPNet report.json is the canonical source for "
            "development/test distributions, scaffold evidence, and DataSAIL leakage."
        ),
    }


def _plot_fold_maxima(
    result: ProtocolSimilarityDiagnostics | CrossValidationSimilarityDiagnostics,
    destination: Path,
) -> None:
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 600, "font.size": 10})
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    values = [
        fold.per_validation["max_source_tanimoto"].to_numpy(dtype=float) for fold in result.folds
    ]
    axis.boxplot(values, tick_labels=[str(fold.fold) for fold in result.folds], showfliers=False)
    axis.axhline(
        result.threshold, color="#c44e52", linestyle="--", linewidth=1, label="report threshold"
    )
    axis.set(
        xlabel="CV fold",
        ylabel="Validation molecule max train Tanimoto",
        title=f"{result.protocol}: train/validation similarity",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def _plot_test_maxima(result: ProtocolSimilarityDiagnostics, destination: Path) -> None:
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 600, "font.size": 10})
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    values = result.test.per_target["max_source_tanimoto"].to_numpy(dtype=float)
    axis.hist(values, bins=30, color="#4c72b0", edgecolor="white")
    axis.axvline(
        result.threshold, color="#c44e52", linestyle="--", linewidth=1, label="report threshold"
    )
    axis.set(
        xlabel="Test molecule max development Tanimoto",
        ylabel="Count",
        title=f"{result.protocol}: development/test similarity",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def _plot_fold_ecdf(
    result: ProtocolSimilarityDiagnostics | CrossValidationSimilarityDiagnostics,
    destination: Path,
) -> None:
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 600, "font.size": 10})
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for fold in result.folds:
        values = np.sort(fold.pairwise_similarities)
        axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=f"fold {fold.fold}")
    axis.set(
        xlabel="Train/validation pair Tanimoto",
        ylabel="ECDF",
        title=f"{result.protocol}: CV pairwise similarity",
    )
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
