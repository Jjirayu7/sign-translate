from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

try:
    from phase1_sign_recognition.sign_translate_phase1.landmarks import FEATURE_COUNT, HandLandmarkExtractor
except ModuleNotFoundError:
    from sign_translate_phase1.landmarks import FEATURE_COUNT, HandLandmarkExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect sign gesture landmark samples.")
    parser.add_argument("--label", required=True, help="Gesture label, for example hello or thanks.")
    parser.add_argument("--output", default="data/raw/sign_landmarks.csv", help="CSV output path.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--samples", type=int, default=120, help="Number of samples to capture.")
    return parser.parse_args()


def ensure_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", *[f"f_{index}" for index in range(FEATURE_COUNT)]])


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    ensure_header(output_path)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    extractor = HandLandmarkExtractor()
    captured = 0
    is_recording = False

    print("Controls: press SPACE to start/pause capture, q to quit.")
    print(f"Collecting label={args.label!r}, target samples={args.samples}")

    try:
        with output_path.open("a", newline="") as file:
            writer = csv.writer(file)

            while captured < args.samples:
                ok, frame = capture.read()
                if not ok:
                    break

                result = extractor.process(frame)
                status = "RECORDING" if is_recording else "PAUSED"

                if is_recording and result.features is not None:
                    writer.writerow([args.label, *result.features.tolist()])
                    captured += 1

                cv2.putText(
                    result.annotated_frame,
                    f"{status} {args.label}: {captured}/{args.samples}",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if is_recording else (0, 200, 255),
                    2,
                )
                cv2.imshow("Collect sign dataset", result.annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    is_recording = not is_recording
    finally:
        extractor.close()
        capture.release()
        cv2.destroyAllWindows()

    print(f"Saved {captured} samples for label={args.label!r} to {output_path}")


if __name__ == "__main__":
    main()
