from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

try:
    from phase1_sign_recognition.sign_translate_phase1.landmarks import FEATURE_COUNT
except ModuleNotFoundError:
    from sign_translate_phase1.landmarks import FEATURE_COUNT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a landmark-based sign classifier.")
    parser.add_argument("--input", default="data/raw/sign_landmarks.csv", help="Input CSV path.")
    parser.add_argument("--model", default="models/sign_classifier.joblib", help="Output model path.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--estimators", type=int, default=250, help="Number of RandomForest trees.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    model_path = Path(args.model)

    data = pd.read_csv(input_path)
    if "label" not in data.columns:
        raise ValueError("Input CSV must contain a label column")

    feature_columns = [f"f_{index}" for index in range(FEATURE_COUNT)]
    missing_columns = [column for column in feature_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Input CSV is missing feature columns: {missing_columns[:5]}")

    features = data[feature_columns].apply(pd.to_numeric, errors="raise")
    labels = data["label"].astype(str)

    label_counts = labels.value_counts()
    if len(label_counts) < 2:
        raise ValueError("Train at least two labels before fitting a classifier")
    if label_counts.min() < 2:
        raise ValueError("Each label needs at least two samples for stratified validation")

    n_classes = len(label_counts)
    test_count = math.ceil(len(labels) * args.test_size)
    train_count = len(labels) - test_count
    if test_count < n_classes or train_count < n_classes:
        raise ValueError(
            "Validation split is too small for the number of labels. "
            "Collect more samples or increase --test-size."
        )

    if "source_id" in data.columns and data["source_id"].nunique() >= 2:
        n_splits = min(5, int(data["source_id"].nunique()))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        train_indices, test_indices = next(splitter.split(features, labels, groups=data["source_id"]))
        x_train, x_test = features.iloc[train_indices], features.iloc[test_indices]
        y_train, y_test = labels.iloc[train_indices], labels.iloc[test_indices]
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=args.test_size,
            random_state=42,
            stratify=labels,
        )

    model = RandomForestClassifier(
        n_estimators=args.estimators,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions, zero_division=0))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "labels": sorted(labels.unique().tolist()),
        },
        model_path,
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
