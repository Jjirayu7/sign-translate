from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from phase1_sign_recognition.sign_translate_phase1.landmarks import normalize_landmarks
except ModuleNotFoundError:
    from sign_translate_phase1.landmarks import normalize_landmarks


OUTPUT_FEATURE_COUNT = 63
HAND_COLUMNS = {
    "lh": [f"lh_{axis}{index}" for index in range(21) for axis in ("x", "y", "z")],
    "rh": [f"rh_{axis}{index}" for index in range(21) for axis in ("x", "y", "z")],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TSL-51 one-hand landmark files to the Phase 1 single-frame CSV format."
    )
    parser.add_argument("dataset_root", help="Path to the cloned TSL-51 dataset directory.")
    parser.add_argument("--output", default="data/raw/tsl51_landmarks.csv", help="Output CSV path.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap for quick tests. 0 means no cap.")
    parser.add_argument(
        "--label-key",
        default="",
        help="Optional explicit label key when landmark records are dict-like.",
    )
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="Allow loading .pkl/.pickle and pickled NumPy object arrays from trusted local datasets only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    metadata = load_user_sign_metadata(root)
    rows: list[dict[str, Any]] = []
    for file_path in iter_candidate_files(root, include_pickle=args.allow_unsafe_pickle):
        metadata_record = metadata.get(file_path.relative_to(root).as_posix())
        if file_path.suffix.lower() == ".csv" and has_tsl_landmark_columns(file_path):
            label = metadata_record["label"] if metadata_record else infer_label_from_filename(file_path)
            source_id = metadata_record.get("video_id", file_path.stem) if metadata_record else file_path.stem
            for frame in load_tsl_csv_frames(file_path):
                features = extract_63_features(frame)
                if features is None:
                    continue
                rows.append({"label": label, "source_id": source_id, **feature_row(features)})
                if args.max_rows and len(rows) >= args.max_rows:
                    write_output(rows, Path(args.output))
                    return
            continue

        for label, frames in load_labeled_sequences(
            file_path,
            root,
            args.label_key or None,
            allow_unsafe_pickle=args.allow_unsafe_pickle,
        ):
            for frame in frames:
                features = extract_63_features(frame)
                if features is None:
                    continue
                rows.append({"label": label, **feature_row(features)})
                if args.max_rows and len(rows) >= args.max_rows:
                    write_output(rows, Path(args.output))
                    return

    if not rows:
        raise RuntimeError(
            "No convertible landmark rows found. Run inspect_tsl51.py on the dataset root "
            "and check the landmark/metadata format."
        )

    write_output(rows, Path(args.output))


def iter_candidate_files(root: Path, include_pickle: bool = False) -> Iterable[Path]:
    extensions = {".csv", ".json", ".jsonl", ".npy", ".npz"}
    if include_pickle:
        extensions.update({".pkl", ".pickle"})

    user_sign_root = root / "landmarks" / "user_sign"
    if user_sign_root.exists():
        search_root = user_sign_root
    else:
        search_root = root / "landmarks" if (root / "landmarks").exists() else root
    for path in search_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def load_user_sign_metadata(root: Path) -> dict[str, dict[str, str]]:
    metadata_path = root / "metadata" / "user_sign_metadata.csv"
    if not metadata_path.exists():
        return {}

    data = pd.read_csv(metadata_path, dtype=str).fillna("")
    result: dict[str, dict[str, str]] = {}
    for record in data.to_dict("records"):
        landmark_path = record.get("landmark_path", "").replace("\\", "/")
        if landmark_path:
            result[landmark_path] = {
                "label": record.get("sign_clean", ""),
                "video_id": record.get("video_id", ""),
            }
    return result


def has_tsl_landmark_columns(path: Path) -> bool:
    columns = set(pd.read_csv(path, nrows=0).columns)
    return set(HAND_COLUMNS["lh"]).issubset(columns) or set(HAND_COLUMNS["rh"]).issubset(columns)


def load_tsl_csv_frames(path: Path) -> Iterable[list[float]]:
    data = pd.read_csv(path)
    available_hands = [hand for hand, columns in HAND_COLUMNS.items() if set(columns).issubset(data.columns)]
    for _, row in data.iterrows():
        for hand in ("rh", "lh"):
            if hand not in available_hands:
                continue
            values = pd.to_numeric(row[HAND_COLUMNS[hand]], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).all() and not np.allclose(values, 0):
                yield values.tolist()
                break


def infer_label_from_filename(path: Path) -> str:
    stem = path.stem
    suffix = stem.rsplit("_", 1)[-1]
    return stem.rsplit("_", 1)[0] if suffix.isdigit() else stem


def feature_row(features: list[float]) -> dict[str, float]:
    return {f"f_{i}": value for i, value in enumerate(features)}


def load_labeled_sequences(
    path: Path,
    root: Path,
    label_key: str | None,
    allow_unsafe_pickle: bool = False,
) -> Iterable[tuple[str, list[Any]]]:
    if path.suffix == ".csv":
        yield from load_csv(path, label_key)
    elif path.suffix == ".json":
        yield from load_records(json.loads(path.read_text()), infer_label_from_path(path, root), label_key)
    elif path.suffix == ".jsonl":
        with path.open() as file:
            for line in file:
                if line.strip():
                    yield from load_records(json.loads(line), infer_label_from_path(path, root), label_key)
    elif path.suffix in {".pkl", ".pickle"}:
        if not allow_unsafe_pickle:
            return
        with path.open("rb") as file:
            yield from load_records(pickle.load(file), infer_label_from_path(path, root), label_key)
    elif path.suffix in {".npy", ".npz"}:
        value = np.load(path, allow_pickle=allow_unsafe_pickle)
        yield infer_label_from_path(path, root), coerce_frames(value)


def load_csv(path: Path, label_key: str | None) -> Iterable[tuple[str, list[Any]]]:
    data = pd.read_csv(path)
    label_column = label_key or find_label_column(data.columns)
    feature_columns = [column for column in data.columns if str(column).startswith("f_")]
    if label_column and len(feature_columns) >= OUTPUT_FEATURE_COUNT:
        for label, group in data.groupby(label_column):
            yield str(label), group[feature_columns[:OUTPUT_FEATURE_COUNT]].to_numpy().tolist()


def load_records(value: Any, fallback_label: str, label_key: str | None) -> Iterable[tuple[str, list[Any]]]:
    if isinstance(value, dict):
        label = extract_label(value, fallback_label, label_key)
        landmarks = find_landmarks(value)
        if landmarks is not None:
            yield label, coerce_frames(landmarks)
        else:
            for child in value.values():
                yield from load_records(child, label, label_key)
    elif isinstance(value, list):
        for item in value:
            yield from load_records(item, fallback_label, label_key)


def extract_label(record: dict[str, Any], fallback: str, label_key: str | None) -> str:
    keys = [label_key] if label_key else []
    keys.extend(["label", "class", "class_name", "sign", "gloss", "word", "text", "name"])
    for key in keys:
        if key and key in record and record[key] not in (None, ""):
            return str(record[key])
    return fallback


def find_landmarks(record: dict[str, Any]) -> Any | None:
    for key in ["landmarks", "landmark", "pose", "hands", "hand_landmarks", "data", "frames"]:
        value = record.get(key)
        if value is not None:
            return value
    return None


def coerce_frames(value: Any) -> list[Any]:
    if isinstance(value, np.lib.npyio.NpzFile):
        frames: list[Any] = []
        for key in value.files:
            frames.extend(coerce_frames(value[key]))
        return frames

    array = np.asarray(value, dtype=object)
    if array.ndim == 1 and array.size >= OUTPUT_FEATURE_COUNT:
        return [array[:OUTPUT_FEATURE_COUNT].astype(float).tolist()]
    if array.ndim == 2 and array.shape[-1] == 3 and array.shape[0] >= 21:
        return [array[:21, :3].astype(float).reshape(-1).tolist()]
    if array.ndim == 2 and array.shape[-1] >= OUTPUT_FEATURE_COUNT:
        return array[:, :OUTPUT_FEATURE_COUNT].astype(float).tolist()
    if array.ndim == 3 and array.shape[-2] >= 21 and array.shape[-1] >= 3:
        return array[:, :21, :3].astype(float).reshape(array.shape[0], -1).tolist()
    return []


def extract_63_features(frame: Any) -> list[float] | None:
    try:
        array = np.asarray(frame, dtype=float)
    except (TypeError, ValueError):
        return None

    if array.ndim == 1 and array.size >= OUTPUT_FEATURE_COUNT:
        return normalize_landmarks(array[:OUTPUT_FEATURE_COUNT].reshape(21, 3)).tolist()
    if array.ndim == 2 and array.shape[0] >= 21 and array.shape[1] >= 3:
        return normalize_landmarks(array[:21, :3]).tolist()
    return None


def find_label_column(columns: Iterable[Any]) -> str | None:
    normalized = {str(column).lower(): str(column) for column in columns}
    for key in ["label", "class", "class_name", "sign", "gloss", "word", "text"]:
        if key in normalized:
            return normalized[key]
    return None


def infer_label_from_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    parts = relative.parts
    return parts[-2] if len(parts) > 1 else path.stem


def write_output(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    data.to_csv(output_path, index=False)
    print(f"Wrote {len(data)} rows, {data['label'].nunique()} labels to {output_path}")


if __name__ == "__main__":
    main()
