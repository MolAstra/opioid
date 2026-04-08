#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


INPUT_CSV = Path("datasets/opioid.csv")
OUTPUT_DIR = Path("datasets/cv")
TEST_FRAC = 0.1
N_FOLDS = 5
SEED = 2026
STATS_JSON = OUTPUT_DIR / "split_stats.json"


def load_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def positive_ratio(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    positives = sum(int(row["label"]) for row in rows)
    return positives / len(rows)


def print_split_stats(name: str, rows: list[dict[str, str]]) -> None:
    print(f"{name}: n={len(rows)}, pos_ratio={positive_ratio(rows):.4f}")


def split_stats(rows: list[dict[str, str]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "pos_ratio": round(positive_ratio(rows), 6),
    }


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_test_split(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    scaffold_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    empty_scaffold_rows: list[dict[str, str]] = []
    for row in rows:
        key = scaffold_key(row["smiles"])
        if key:
            scaffold_groups[key].append(row)
        else:
            empty_scaffold_rows.append(row)

    groups = list(scaffold_groups.values())
    random.Random(SEED).shuffle(groups)

    target_test_size = round(len(rows) * TEST_FRAC)
    test_rows: list[dict[str, str]] = []
    train_valid_rows: list[dict[str, str]] = list(empty_scaffold_rows)

    for group in groups:
        if len(test_rows) < target_test_size:
            test_rows.extend(group)
        else:
            train_valid_rows.extend(group)

    return train_valid_rows, test_rows


def make_folds(
    rows: list[dict[str, str]],
) -> list[tuple[list[dict[str, str]], list[dict[str, str]]]]:
    shuffled = rows[:]
    random.Random(SEED).shuffle(shuffled)

    fold_sizes = [len(shuffled) // N_FOLDS] * N_FOLDS
    for i in range(len(shuffled) % N_FOLDS):
        fold_sizes[i] += 1

    folds: list[list[dict[str, str]]] = []
    start = 0
    for fold_size in fold_sizes:
        end = start + fold_size
        folds.append(shuffled[start:end])
        start = end

    split_pairs: list[tuple[list[dict[str, str]], list[dict[str, str]]]] = []
    for i in range(N_FOLDS):
        valid_rows = folds[i]
        train_rows: list[dict[str, str]] = []
        for j in range(N_FOLDS):
            if j != i:
                train_rows.extend(folds[j])
        split_pairs.append((train_rows, valid_rows))

    return split_pairs


def main() -> None:
    rows = load_rows()
    fieldnames = list(rows[0].keys())
    stats: dict[str, dict[str, float | int]] = {}

    train_valid_rows, test_rows = make_test_split(rows)
    write_rows(OUTPUT_DIR / "test.csv", test_rows, fieldnames)
    stats["all"] = split_stats(rows)
    stats["train_valid"] = split_stats(train_valid_rows)
    stats["test"] = split_stats(test_rows)
    print_split_stats("all", rows)
    print_split_stats("train_valid", train_valid_rows)
    print_split_stats("test", test_rows)

    for fold_idx, (train_rows, valid_rows) in enumerate(make_folds(train_valid_rows)):
        fold_dir = OUTPUT_DIR / f"fold_{fold_idx}"
        write_rows(fold_dir / "train.csv", train_rows, fieldnames)
        write_rows(fold_dir / "valid.csv", valid_rows, fieldnames)
        stats[f"fold_{fold_idx}"] = {
            "train": split_stats(train_rows),
            "valid": split_stats(valid_rows),
        }
        print_split_stats(f"fold_{fold_idx}/train", train_rows)
        print_split_stats(f"fold_{fold_idx}/valid", valid_rows)

    STATS_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"saved stats: {STATS_JSON}")


if __name__ == "__main__":
    main()
