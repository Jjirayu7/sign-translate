import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from phase1_sign_recognition.convert_tsl51_landmarks import (
    HAND_COLUMNS,
    extract_63_features,
    load_tsl_csv_frames,
    load_user_sign_metadata,
)
from phase1_sign_recognition.sign_translate_phase1.landmarks import normalize_landmarks


class PipelineTests(unittest.TestCase):
    def test_normalize_landmarks_returns_wrist_origin_and_63_features(self):
        points = [(float(i), float(i * 2), float(i * 3)) for i in range(21)]
        features = normalize_landmarks(points)
        self.assertEqual(features.shape, (63,))
        np.testing.assert_array_equal(features[:3], np.zeros(3))

    def test_tsl_csv_loader_reads_right_hand_and_skips_empty_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            columns = ["frame", *HAND_COLUMNS["lh"], *HAND_COLUMNS["rh"]]
            row = {column: 0.0 for column in columns}
            row["frame"] = 1
            for index, column in enumerate(HAND_COLUMNS["rh"]):
                row[column] = float(index + 1)
            pd.DataFrame([row, {**row, "frame": 2, **{column: 0.0 for column in HAND_COLUMNS["rh"]}}]).to_csv(path, index=False)

            frames = list(load_tsl_csv_frames(path))

            self.assertEqual(len(frames), 1)
            self.assertEqual(len(frames[0]), 63)
            self.assertEqual(len(extract_63_features(frames[0])), 63)

    def test_metadata_maps_landmark_path_to_clean_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            pd.DataFrame(
                [{
                    "landmark_path": "landmarks/user_sign/sample.csv",
                    "sign_clean": "สวัสดี",
                    "video_id": "sample_0",
                }]
            ).to_csv(root / "metadata" / "user_sign_metadata.csv", index=False)

            metadata = load_user_sign_metadata(root)

            self.assertEqual(metadata["landmarks/user_sign/sample.csv"]["label"], "สวัสดี")
            self.assertEqual(metadata["landmarks/user_sign/sample.csv"]["video_id"], "sample_0")


if __name__ == "__main__":
    unittest.main()
