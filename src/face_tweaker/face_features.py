"""MediaPipe face measurements for one video frame.

MediaPipe Face Landmarker gives the points that place the ArcFace crop
(FIVE_POINTS), and the head rotation, which gives yaw and pitch. ArcFace
itself (InsightFace buffalo_l, w600k_r50) runs in bareface. On our clips it
follows the sliders far better than MediaPipe points or a 3DDFA_V2 face model
fit.

The MediaPipe model lives in data/models.
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode

MODELS = Path(__file__).resolve().parents[2] / "data" / "models"

# MediaPipe points that stand in for the 5 points ArcFace aligns on, in the
# order of ArcFace's template: the eye on the image left, the eye on the image
# right, the nose tip, the mouth corner on the image left, the one on the right.
# 468 and 473 are the iris centers.
FIVE_POINTS = [468, 473, 1, 61, 291]


class Landmarks:
    """MediaPipe Face Landmarker in video mode. Feed frames in order."""

    def __init__(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """(points, yaw, pitch), or None if no face.

        Points are (478, 3) in pixels, with z on the same scale as x. Yaw and
        pitch are in degrees, from the direction the face points. On the idle
        clip this yaw tracks a 3DDFA_V2 fit's yaw with correlation 0.99 and
        1.3 degrees of scatter, about 5 degrees apart at zero.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        if not result.face_landmarks:
            return None
        points = np.array([[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]], np.float32)
        forward = np.array(result.facial_transformation_matrixes[0])[:3, 2]
        yaw = np.degrees(np.arctan2(forward[0], forward[2]))
        pitch = np.degrees(np.arctan2(-forward[1], np.hypot(forward[0], forward[2])))
        return points, float(yaw), float(pitch)

    def close(self):
        self.landmarker.close()
