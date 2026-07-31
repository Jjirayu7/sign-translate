from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

try:
    from phase1_sign_recognition.sign_translate_phase1.landmarks import FEATURE_COUNT, HandLandmarkExtractor
except ModuleNotFoundError:
    from sign_translate_phase1.landmarks import FEATURE_COUNT, HandLandmarkExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime sign prediction from camera.")
    parser.add_argument("--model", default="models/sign_classifier.joblib", help="Model path.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Minimum confidence to show a label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = joblib.load(Path(args.model))
    model = artifact["model"]
    feature_columns = artifact.get("feature_columns") or [f"f_{index}" for index in range(FEATURE_COUNT)]
    if len(feature_columns) != FEATURE_COUNT:
        raise ValueError(f"Model artifact has {len(feature_columns)} feature columns, expected {FEATURE_COUNT}")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    extractor = HandLandmarkExtractor()
    print("Controls: press q to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = extractor.process(frame)
            label = "No hand"
            confidence = 0.0

            if result.features is not None:
                sample = pd.DataFrame([result.features], columns=feature_columns)
                probabilities = model.predict_proba(sample)[0]
                best_index = int(np.argmax(probabilities))
                confidence = float(probabilities[best_index])
                label = model.classes_[best_index] if confidence >= args.threshold else "Unknown"

            cv2.putText(
                result.annotated_frame,
                f"{label} ({confidence:.2f})",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Sign realtime prediction", result.annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        extractor.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
