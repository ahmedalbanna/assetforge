"""
Character Feature Detection — detects face, eyes, mouth, and clothing regions.

Provides feature-aware region extraction for enhancing SVG vectorization
quality on character images (anime, realistic, or stylized art).

Approach:
1. Face detection via OpenCV DNN (Caffe SSD) or Haar cascades
2. Within face bounding box, refine eye/mouth regions using:
   - Adaptive thresholding for dark features
   - Contour analysis for shape-based filtering
   - Geometric heuristics (eyes in upper half, mouth in lower)
3. Generate binary semantic masks for feature-preserving vectorization

Supports both realistic and anime/manga style characters.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("assetforge")

# Face landmark indices for reference (MediaPipe-compatible if used):
# Left eye: 33-133, Right eye: 362-263, Mouth: 61-291, Nose: 1-19
# We use geometric heuristics instead of landmarks for zero-dependency operation.


class DetectionMethod(str, Enum):
    """Face detection backend selection."""

    AUTO = "auto"  # Try DNN first, fall back to Haar
    DNN = "dnn"  # OpenCV DNN SSD face detector
    HAAR = "haar"  # Haar cascade classifier
    HEURISTIC = "heuristic"  # Assume full image is a face (no detection)


@dataclass
class CharacterFeatures:
    """Detected character features for enhanced vectorization.

    Attributes:
        face_detected: Whether any face was detected.
        face_bbox: Bounding box of the primary face (x, y, w, h) in pixels.
        feature_masks: Dict mapping feature name -> binary uint8 mask (0/255).
                       Keys: 'face', 'left_eye', 'right_eye', 'mouth',
                             'left_eyebrow', 'right_eyebrow', 'nose'.
        face_landmarks: Optional 2D landmark points (Nx2) if available.
        face_confidence: Detection confidence score (0-1).
        num_faces: Number of faces detected (primary = first/ largest).
        eye_regions: List of (x, y, w, h) for detected eyes.
        mouth_region: (x, y, w, h) for detected mouth, or None.
    """

    face_detected: bool = False
    face_bbox: Optional[tuple[int, int, int, int]] = None
    feature_masks: dict[str, np.ndarray] = field(default_factory=dict)
    face_landmarks: Optional[np.ndarray] = None
    face_confidence: float = 0.0
    num_faces: int = 0
    eye_regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    mouth_region: Optional[tuple[int, int, int, int]] = None

    def has_feature(self, name: str) -> bool:
        """Check if a named feature mask exists and has non-zero pixels."""
        mask = self.feature_masks.get(name)
        return bool(mask is not None and mask.sum() > 0)

    def get_feature_bbox(self, name: str) -> Optional[tuple[int, int, int, int]]:
        """Get the bounding box of a feature mask."""
        mask = self.feature_masks.get(name)
        if mask is None or (mask > 0).sum() == 0:
            return None
        ys, xs = np.where(mask > 0)
        x, y = xs.min(), ys.min()
        w = xs.max() - x + 1
        h = ys.max() - y + 1
        return (int(x), int(y), int(w), int(h))


def _download_dnn_model() -> tuple[str, str]:
    """Download DNN face detector model files if not present.

    Returns:
        Tuple of (prototxt_path, model_path).
    """
    import os
    import urllib.request

    model_dir = os.path.expanduser("~/.cache/assetforge/models")
    os.makedirs(model_dir, exist_ok=True)

    proto_path = os.path.join(model_dir, "deploy.prototxt")
    model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(proto_path):
        logger.info("Downloading DNN face detector prototxt...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",  # noqa: E501
            proto_path,
        )
    if not os.path.exists(model_path):
        logger.info("Downloading DNN face detector model (~10 MB)...")
        urllib.request.urlretrieve(
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",  # noqa: E501
            model_path,
        )
    return proto_path, model_path


def _detect_faces_dnn(image_rgb: np.ndarray, min_confidence: float = 0.5) -> list[dict]:
    """Detect faces using OpenCV DNN SSD detector.

    Args:
        image_rgb: RGB uint8 image.
        min_confidence: Minimum detection confidence.

    Returns:
        List of dicts with 'bbox' (x, y, w, h) and 'confidence'.
    """
    h, w = image_rgb.shape[:2]
    proto_path, model_path = _download_dnn_model()
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    # Construct input blob
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_rgb, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < min_confidence:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        faces.append(
            {
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "confidence": confidence,
            }
        )
    return faces


def _detect_faces_haar(
    image_gray: np.ndarray,
    cascade_path: str = "",
) -> list[dict]:
    """Detect faces using Haar cascade classifier.

    Args:
        image_gray: Grayscale uint8 image.
        cascade_path: Path to Haar cascade XML. Empty = use default.

    Returns:
        List of dicts with 'bbox' (x, y, w, h).
    """
    if not cascade_path:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    rects = cascade.detectMultiScale(
        image_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    return [{"bbox": (x, y, w, h), "confidence": 1.0} for (x, y, w, h) in rects]


def _detect_faces(
    image_rgb: np.ndarray,
    method: DetectionMethod = DetectionMethod.AUTO,
    min_confidence: float = 0.5,
) -> list[dict]:
    """Detect faces using the specified method.

    Args:
        image_rgb: RGB uint8 image.
        method: Detection backend.
        min_confidence: Minimum confidence for DNN detections.

    Returns:
        List of dicts with 'bbox' and 'confidence'.
    """
    if method == DetectionMethod.HEURISTIC:
        h, w = image_rgb.shape[:2]
        return [{"bbox": (0, 0, w, h), "confidence": 1.0}]

    if method in (DetectionMethod.AUTO, DetectionMethod.DNN):
        try:
            faces = _detect_faces_dnn(image_rgb, min_confidence)
            if faces:
                return faces
            if method == DetectionMethod.DNN:
                return faces  # Return empty if DNN explicitly requested
        except Exception as e:
            logger.warning("DNN face detection error, falling back to Haar: %s", e)

    # Fallback to Haar
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return _detect_faces_haar(image_gray)


def _refine_eye_regions(
    image_rgb: np.ndarray,
    face_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    """Refine eye positions within face region using image processing.

    Uses adaptive thresholding on the upper half of the face to find
    dark regions (eyes), then filters by shape and position.

    Args:
        image_rgb: RGB uint8 image.
        face_bbox: Face bounding box (x, y, w, h).

    Returns:
        List of eye bounding boxes (x, y, w, h), typically 0-2.
    """
    fx, fy, fw, fh = face_bbox
    if fw < 20 or fh < 20:
        return []

    # Focus on upper 60% of face for eyes
    eye_roi_y = fy + int(fh * 0.05)
    eye_roi_h = int(fh * 0.55)
    eye_roi = image_rgb[eye_roi_y : eye_roi_y + eye_roi_h, fx : fx + fw]
    if eye_roi.shape[0] < 5 or eye_roi.shape[1] < 5:
        return []

    # Convert to grayscale and threshold to find dark regions
    gray = cv2.cvtColor(eye_roi, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold to find dark spots (eyes, eyebrows)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        5,
    )

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours of dark regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by expected eye shape and position
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10 or area > eye_roi_h * eye_roi.shape[1] * 0.3:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        # Eyes are typically wider than tall (aspect > 1.5) and not too extreme
        if aspect < 0.8 or aspect > 5.0:
            continue
        # Map back to original image coordinates
        abs_x = fx + x
        abs_y = eye_roi_y + y
        candidates.append((abs_x, abs_y, w, h, area))

    # Sort by area descending, take top 2 that are horizontally separated
    candidates.sort(key=lambda c: c[4], reverse=True)

    eyes = []
    for cand in candidates:
        cx, cy, cw, ch = cand[:4]
        # Check if this candidate overlaps with existing eyes
        overlap = False
        for ex, ey, ew, eh in eyes:
            if _bbox_iou((cx, cy, cw, ch), (ex, ey, ew, eh)) > 0.3:
                overlap = True
                break
        if not overlap:
            eyes.append((cx, cy, cw, ch))
        if len(eyes) >= 2:
            break

    # If we only found 0-1 eyes, try a more aggressive approach
    if len(eyes) < 2:
        # Use simpler thresholding
        _, simple_thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        simple_thresh = cv2.morphologyEx(simple_thresh, cv2.MORPH_CLOSE, kernel)
        contours2, _ = cv2.findContours(
            simple_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        extra_candidates = []
        for cnt in contours2:
            area = cv2.contourArea(cnt)
            if area < 15 or area > eye_roi_h * eye_roi.shape[1] * 0.3:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if aspect < 0.8 or aspect > 5.0:
                continue
            abs_x = fx + x
            abs_y = eye_roi_y + y
            # Check overlap with existing
            overlap = False
            for ex, ey, ew, eh in eyes:
                if _bbox_iou((abs_x, abs_y, w, h), (ex, ey, ew, eh)) > 0.3:
                    overlap = True
                    break
            if not overlap:
                extra_candidates.append((abs_x, abs_y, w, h))
            if len(eyes) + len(extra_candidates) >= 2:
                break
        eyes.extend(extra_candidates[: 2 - len(eyes)])

    return eyes


def _refine_mouth_region(
    image_rgb: np.ndarray,
    face_bbox: tuple[int, int, int, int],
) -> Optional[tuple[int, int, int, int]]:
    """Find mouth region within face using image processing.

    Looks in the lower 40% of the face for a horizontally elongated
    dark/red region.

    Args:
        image_rgb: RGB uint8 image.
        face_bbox: Face bounding box (x, y, w, h).

    Returns:
        Mouth bounding box (x, y, w, h) or None.
    """
    fx, fy, fw, fh = face_bbox
    if fw < 20 or fh < 20:
        return None

    # Focus on lower 45% of face
    mouth_roi_y = fy + int(fh * 0.50)
    mouth_roi_h = int(fh * 0.45)
    mouth_roi = image_rgb[mouth_roi_y : mouth_roi_y + mouth_roi_h, fx : fx + fw]
    if mouth_roi.shape[0] < 5 or mouth_roi.shape[1] < 5:
        return None

    # For mouth: look for dark regions (closed mouth) or red regions (open/lips)
    gray = cv2.cvtColor(mouth_roi, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Also check red channel for lip detection
    red_channel = mouth_roi[:, :, 0].astype(np.float32)
    green_channel = mouth_roi[:, :, 1].astype(np.float32)
    red_ratio = red_channel / (green_channel + 1e-6)

    # Combine: dark regions + red regions
    _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    _, red_mask = cv2.threshold(
        (red_ratio * 50).astype(np.uint8), 30, 255, cv2.THRESH_BINARY
    )

    combined = cv2.bitwise_or(dark_mask, red_mask)
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    roi_area = mouth_roi.shape[0] * mouth_roi.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > roi_area * 0.7:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        # Mouth is typically wider than tall
        if aspect < 0.5 or aspect > 6.0:
            continue
        # Prefer regions centered horizontally and in lower portion
        center_x_ratio = (x + w / 2) / mouth_roi.shape[1]
        if center_x_ratio < 0.15 or center_x_ratio > 0.85:
            continue
        if area > best_area:
            best_area = area
            best = (fx + x, mouth_roi_y + y, w, h)

    return best


def _generate_face_mask(
    image_shape: tuple[int, int],
    face_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Generate an elliptical face mask within the face bounding box.

    Args:
        image_shape: (height, width) of the image.
        face_bbox: Face bounding box (x, y, w, h).

    Returns:
        Binary uint8 mask (0/255).
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    fx, fy, fw, fh = face_bbox
    # Elliptical face shape
    center = (fx + fw // 2, fy + fh // 2)
    axes = (fw // 2, int(fh * 0.80) // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _generate_eye_mask(
    image_shape: tuple[int, int],
    eye_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Generate an elliptical eye mask.

    Args:
        image_shape: (height, width) of the image.
        eye_bbox: Eye bounding box (x, y, w, h).

    Returns:
        Binary uint8 mask (0/255).
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    ex, ey, ew, eh = eye_bbox
    center = (ex + ew // 2, ey + eh // 2)
    axes = (ew // 2, eh // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _generate_mouth_mask(
    image_shape: tuple[int, int],
    mouth_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Generate a mouth mask (ellipse).

    Args:
        image_shape: (height, width) of the image.
        mouth_bbox: Mouth bounding box (x, y, w, h).

    Returns:
        Binary uint8 mask (0/255).
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mx, my, mw, mh = mouth_bbox
    center = (mx + mw // 2, my + mh // 2)
    axes = (mw // 2, mh // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _bbox_iou(
    bbox1: tuple[int, int, int, int],
    bbox2: tuple[int, int, int, int],
) -> float:
    """Compute IoU of two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    union = area1 + area2 - inter
    return inter / max(union, 1)


def detect_character_features(
    image: np.ndarray,
    method: DetectionMethod = DetectionMethod.AUTO,
    min_face_size: int = 30,
    min_confidence: float = 0.4,
) -> CharacterFeatures:
    """Detect character facial features for enhanced vectorization.

    Detects faces using OpenCV (DNN or Haar), then refines eye and mouth
    regions within each face using adaptive image processing techniques.

    Args:
        image: RGB uint8 array, shape (H, W, 3).
        method: Face detection backend to use.
        min_face_size: Minimum face dimension in pixels.
        min_confidence: Minimum detection confidence (DNN only).

    Returns:
        CharacterFeatures dataclass with detected regions and masks.
    """
    h, w = image.shape[:2]
    features = CharacterFeatures()

    # Ensure RGB
    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image_rgb = image[:, :, :3]
    else:
        image_rgb = image

    # Step 1: Detect faces
    detected_faces = _detect_faces(image_rgb, method, min_confidence)
    if not detected_faces:
        logger.debug("No faces detected.")
        return features

    # Step 2: Pick the largest face (primary)
    primary = max(detected_faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
    fx, fy, fw, fh = primary["bbox"]

    if fw < min_face_size or fh < min_face_size:
        logger.debug("Face too small: %dx%d", fw, fh)
        return features

    features.face_detected = True
    features.face_bbox = (fx, fy, fw, fh)
    features.face_confidence = primary.get("confidence", 1.0)
    features.num_faces = len(detected_faces)

    # Step 3: Generate face mask
    features.feature_masks["face"] = _generate_face_mask((h, w), (fx, fy, fw, fh))

    # Step 4: Detect eye regions
    eye_bboxes = _refine_eye_regions(image_rgb, (fx, fy, fw, fh))
    # Sort eyes left-to-right by x coordinate
    eye_bboxes.sort(key=lambda e: e[0])
    features.eye_regions = eye_bboxes

    for i, ebbox in enumerate(eye_bboxes[:2]):
        name = "left_eye" if i == 0 else "right_eye"
        features.feature_masks[name] = _generate_eye_mask((h, w), ebbox)

    # Step 5: Detect mouth region
    mouth_bbox = _refine_mouth_region(image_rgb, (fx, fy, fw, fh))
    if mouth_bbox is not None:
        features.mouth_region = mouth_bbox
        features.feature_masks["mouth"] = _generate_mouth_mask((h, w), mouth_bbox)

    logger.debug(
        "Character features detected: face=%s eyes=%d mouth=%s",
        features.face_detected,
        len(features.eye_regions),
        features.mouth_region is not None,
    )

    return features


def enhance_feature_masks(
    image: np.ndarray,
    features: CharacterFeatures,
    dilation_kernel: int = 3,
) -> CharacterFeatures:
    """Post-process feature masks for better coverage.

    Dilates feature masks to create soft transition regions that help
    preserve detail gradients during vectorization.

    Args:
        image: RGB uint8 image (used for size reference).
        features: Detected character features.
        dilation_kernel: Kernel size for morph dilation.

    Returns:
        Updated CharacterFeatures with enhanced masks.
    """
    if not features.face_detected:
        return features

    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)

    for name in list(features.feature_masks.keys()):
        mask = features.feature_masks[name]
        if mask is not None:
            dilated = cv2.dilate(mask, kernel, iterations=1)
            features.feature_masks[name] = dilated

    return features


def visualize_features(
    image: np.ndarray,
    features: CharacterFeatures,
) -> np.ndarray:
    """Draw detected features on image for debugging.

    Args:
        image: RGB uint8 image.
        features: Detected character features.

    Returns:
        BGR uint8 image with visualized features.
    """
    vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if features.face_bbox:
        fx, fy, fw, fh = features.face_bbox
        cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

    for name, mask in features.feature_masks.items():
        color = {
            "face": (255, 255, 0),
            "left_eye": (0, 0, 255),
            "right_eye": (255, 0, 0),
            "mouth": (0, 255, 255),
        }.get(name, (128, 128, 128))
        # Draw contour of mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)

    for i, ebbox in enumerate(features.eye_regions):
        ex, ey, ew, eh = ebbox
        label = f"Eye_{i}"
        cv2.putText(
            vis, label, (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
        )

    if features.mouth_region:
        mx, my, mw, mh = features.mouth_region
        cv2.rectangle(vis, (mx, my), (mx + mw, my + mh), (0, 255, 255), 2)
        cv2.putText(
            vis, "Mouth", (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
        )

    return vis
