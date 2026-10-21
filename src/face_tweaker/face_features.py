"""Face measurements for one video frame.

- ArcFace (InsightFace buffalo_l, w600k_r50): a 512-number identity embedding,
  built for face recognition. On our clips it follows the sliders far better
  than MediaPipe points or a 3DDFA_V2 face model fit.
- MediaPipe Face Landmarker: places the ArcFace crop from its points, and
  reports the head rotation, which gives yaw and pitch.

Model files live in data/models.
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode

MODELS = Path(__file__).resolve().parents[2] / "data" / "models"

# MediaPipe points that stand in for the 5 points ArcFace aligns on, in the
# order of ARCFACE_TEMPLATE: the eye on the image left, the eye on the image
# right, the nose tip, the mouth corner on the image left, the one on the right.
# 468 and 473 are the iris centers.
FIVE_POINTS = [468, 473, 1, 61, 291]

# Where ArcFace expects those 5 points in its 112 x 112 input. This is the
# standard InsightFace alignment template.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


# onnxruntime-gpu finds CUDA and cuDNN in the NVIDIA wheels that its cuda and
# cudnn extras install, but only after this call loads them.
onnxruntime.preload_dlls()


def _onnx_session(path: Path) -> onnxruntime.InferenceSession:
    """Session on the CUDA provider. ArcFace takes 117 ms per face on 2 CPU
    threads and 4 ms on the GPU, so a silent CPU fallback is an error."""
    session = onnxruntime.InferenceSession(str(path), providers=["CUDAExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA provider did not load for {path.name}: {session.get_providers()}")
    return session


def similarity_2d(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2 x 3 matrix of the least-squares similarity transform from src to dst."""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    s = src - mu_s
    d = dst - mu_d
    u, sig, vt = np.linalg.svd(d.T @ s)
    fix = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        fix[1, 1] = -1
    rot = u @ fix @ vt
    scale = (sig * np.diag(fix)).sum() / (s**2).sum()
    m = np.zeros((2, 3))
    m[:, :2] = scale * rot
    m[:, 2] = mu_d - scale * rot @ mu_s
    return m


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


class ArcFace:
    """InsightFace w600k_r50 identity embedding, L2-normalized."""

    def __init__(self):
        self.session = _onnx_session(MODELS / "buffalo_l" / "w600k_r50.onnx")
        self.input = self.session.get_inputs()[0].name

    def crop(self, frame_bgr: np.ndarray, points: np.ndarray) -> np.ndarray:
        m = similarity_2d(points[FIVE_POINTS, :2].astype(np.float64), ARCFACE_TEMPLATE)
        return cv2.warpAffine(frame_bgr, m, (112, 112), borderValue=0)

    def embed(self, frame_bgr: np.ndarray, points: np.ndarray) -> np.ndarray:
        crop = self.crop(frame_bgr, points)
        # RGB, scaled to -1..1, channels first, as InsightFace feeds this model.
        x = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
        x = x.transpose(2, 0, 1)[None]
        e = np.asarray(self.session.run(None, {self.input: x})[0])[0]
        return e / np.linalg.norm(e)
