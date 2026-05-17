import cv2
import numpy as np


# Load OpenCV's pre-trained face detector
_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return _face_cascade


def detect_faces_in_frame(frame):
    """
    Detect faces in a frame using OpenCV.
    Returns list of (x, y, w, h) bounding boxes.
    """
    if frame is None or frame.size == 0:
        return []

    try:
        cascade = _get_face_cascade()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return []

        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    except Exception:
        return []


def draw_faces_on_frame(frame, faces, labels=None, colors=None):
    """
    Draw bounding boxes and labels on frame.
    faces: list of (x, y, w, h)
    labels: list of strings
    colors: list of (B, G, R) tuples
    """
    if frame is None:
        return frame

    result = frame.copy()

    for i, (x, y, w, h) in enumerate(faces):
        color = (0, 255, 0)
        if colors and i < len(colors):
            color = colors[i]

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        if labels and i < len(labels):
            label = labels[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(result, (x, y - text_h - baseline - 5), (x + text_w, y), color, -1)
            cv2.putText(result, label, (x, y - 5), font, font_scale, (0, 0, 0), thickness)

    return result


def crop_face_from_frame(frame, face_box, padding=20):
    """
    Crop a face region from frame with optional padding.
    face_box: (x, y, w, h)
    Returns cropped face image or None.
    """
    if frame is None or frame.size == 0:
        return None

    x, y, w, h = face_box
    h_frame, w_frame = frame.shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_frame, x + w + padding)
    y2 = min(h_frame, y + h + padding)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


def is_valid_face_image(img, min_size=64):
    """Check if an image is large enough to be a valid face image."""
    if img is None or img.size == 0:
        return False
    h, w = img.shape[:2]
    return h >= min_size and w >= min_size
