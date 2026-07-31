from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


LANDMARK_EXTENSIONS = {".csv", ".json", ".jsonl", ".pkl", ".pickle", ".npy", ".npz", ".parquet"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a local TSL-51 dataset clone.")
    parser.add_argument("dataset_root", help="Path to the cloned TSL-51 dataset directory.")
    parser.add_argument("--max-files", type=int, default=40, help="Maximum matching files to preview.")
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="Preview .pkl/.pickle files from trusted local datasets only.",
    )
    return parser.parse_args()


def preview_file(path: Path, allow_unsafe_pickle: bool = False) -> str:
    try:
        if path.suffix == ".csv":
            data = pd.read_csv(path, nrows=3)
            return f"csv shape~{data.shape} columns={list(data.columns[:10])}"
        if path.suffix == ".parquet":
            data = pd.read_parquet(path)
            return f"parquet shape={data.shape} columns={list(data.columns[:10])}"
        if path.suffix == ".json":
            with path.open() as file:
                value = json.load(file)
            return describe_value(value)
        if path.suffix == ".jsonl":
            with path.open() as file:
                first = file.readline()
            return describe_value(json.loads(first)) if first else "empty jsonl"
        if path.suffix in {".pkl", ".pickle"}:
            if not allow_unsafe_pickle:
                return "pickle preview skipped; pass --allow-unsafe-pickle for trusted local files"
            with path.open("rb") as file:
                value = pickle.load(file)
            return describe_value(value)
        if path.suffix in {".npy", ".npz"}:
            return "numpy array file"
    except Exception as exc:
        return f"preview failed: {type(exc).__name__}: {exc}"

    return "no preview available"


def describe_value(value: Any) -> str:
    if isinstance(value, dict):
        return f"dict keys={list(value.keys())[:10]}"
    if isinstance(value, list):
        first = describe_value(value[0]) if value else "empty"
        return f"list len={len(value)} first={first}"
    return type(value).__name__


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    files = [path for path in root.rglob("*") if path.is_file()]
    landmark_files = [path for path in files if path.suffix.lower() in LANDMARK_EXTENSIONS]

    print(f"Dataset root: {root}")
    print(f"Total files: {len(files)}")
    print(f"Candidate metadata/landmark files: {len(landmark_files)}")

    for path in landmark_files[: args.max_files]:
        rel_path = path.relative_to(root)
        print(f"- {rel_path}: {preview_file(path, allow_unsafe_pickle=args.allow_unsafe_pickle)}")


if __name__ == "__main__":
    main()
