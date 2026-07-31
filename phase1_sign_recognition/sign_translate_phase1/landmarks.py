from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np


FEATURE_COUNT = 63


@dataclass
class LandmarkResult:
    features: np.ndarray | None
    annotated_frame: np.ndarray


class HandLandmarkExtractor:
    """Extracts one-hand normalized MediaPipe landmarks from camera frames."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._drawer = mp.solutions.drawing_utils
        self._connections = mp.solutions.hands.HAND_CONNECTIONS

    def close(self) -> None:
        self._hands.close()

    def process(self, frame_bgr: np.ndarray) -> LandmarkResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)
        annotated = frame_bgr.copy()

        if not result.multi_hand_landmarks:
            return LandmarkResult(features=None, annotated_frame=annotated)

        hand_landmarks = result.multi_hand_landmarks[0]
        self._drawer.draw_landmarks(annotated, hand_landmarks, self._connections)
        features = normalize_landmarks(
            (landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark
        )
        return LandmarkResult(features=features, annotated_frame=annotated)


def normalize_landmarks(points: Iterable[tuple[float, float, float]]) -> np.ndarray:
    landmarks = np.array(list(points), dtype=np.float32)
    if landmarks.shape != (21, 3):
        raise ValueError(f"Expected 21 hand landmarks, got shape {landmarks.shape}")

    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist

    scale = np.max(np.linalg.norm(landmarks[:, :2], axis=1))
    if scale > 0:
        landmarks = landmarks / scale

    return landmarks.reshape(FEATURE_COUNT)
