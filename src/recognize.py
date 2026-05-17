import cv2
import numpy as np
import time
from deepface import DeepFace
from src.detect_face import detect_faces_in_frame, crop_face_from_frame
from src.train_embeddings import (
    load_embeddings, extract_embedding_from_array,
    cosine_similarity
)


MODEL_NAME = "Facenet"
RECOGNITION_THRESHOLD = 0.65
UNKNOWN_LABEL = "Unknown"


def recognize_face_in_embedding(face_embedding, embeddings_dict, threshold=RECOGNITION_THRESHOLD):
    """
    Match a face embedding against stored embeddings.
    Returns (name, confidence) tuple.
    """
    if face_embedding is None or not embeddings_dict:
        return UNKNOWN_LABEL, 0.0

    best_name = UNKNOWN_LABEL
    best_sim = 0.0

    for student_name, emb_list in embeddings_dict.items():
        for stored_emb in emb_list:
            try:
                sim = cosine_similarity(face_embedding, stored_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = student_name
            except Exception:
                continue

    if best_sim < threshold:
        return UNKNOWN_LABEL, best_sim

    return best_name, best_sim


def recognize_faces_in_frame(frame, embeddings_dict):
    """
    Detect and recognize all faces in a frame.
    Returns list of dicts: [{name, confidence, box}]
    """
    results = []

    if frame is None or frame.size == 0 or not embeddings_dict:
        return results

    faces = detect_faces_in_frame(frame)

    for face_box in faces:
        x, y, w, h = face_box
        face_crop = crop_face_from_frame(frame, face_box, padding=15)

        if face_crop is None or face_crop.size == 0:
            results.append({
                "name": UNKNOWN_LABEL,
                "confidence": 0.0,
                "box": face_box
            })
            continue

        try:
            # Resize for FaceNet input
            face_resized = cv2.resize(face_crop, (160, 160))
            embedding = extract_embedding_from_array(face_resized)

            if embedding is not None:
                name, confidence = recognize_face_in_embedding(embedding, embeddings_dict)
            else:
                name, confidence = UNKNOWN_LABEL, 0.0

        except Exception:
            name, confidence = UNKNOWN_LABEL, 0.0

        results.append({
            "name": name,
            "confidence": confidence,
            "box": face_box
        })

    return results


def draw_recognition_results(frame, recognition_results, attendance_status=None):
    """
    Draw bounding boxes and recognition info on frame.
    attendance_status: dict {name: status_string}
    Returns annotated frame.
    """
    if frame is None:
        return frame

    result_frame = frame.copy()

    for res in recognition_results:
        name = res["name"]
        confidence = res["confidence"]
        x, y, w, h = res["box"]

        # Choose color based on status
        if name == UNKNOWN_LABEL:
            color = (0, 0, 255)  # Red
            status_text = "Unknown"
        else:
            status = attendance_status.get(name, "") if attendance_status else ""
            if status == "Already Marked":
                color = (0, 165, 255)  # Orange
                status_text = "Already Marked"
            elif status == "Present":
                color = (0, 255, 0)  # Green
                status_text = "Present"
            else:
                color = (255, 255, 0)  # Yellow
                status_text = ""

        # Draw rectangle
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)

        # Build label
        conf_pct = int(confidence * 100)
        label = f"{name} ({conf_pct}%)"
        if status_text:
            label += f" | {status_text}"

        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y - 10, text_h + 5)
        cv2.rectangle(result_frame,
                      (x, label_y - text_h - baseline - 4),
                      (x + text_w + 4, label_y),
                      color, -1)
        cv2.putText(result_frame, label,
                    (x + 2, label_y - baseline - 2),
                    font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(result_frame, label,
                    (x + 2, label_y - baseline - 2),
                    font, font_scale, (255, 255, 255), thickness)

    return result_frame


class FPSCounter:
    """Simple FPS counter."""
    def __init__(self, avg_frames=30):
        self.timestamps = []
        self.avg_frames = avg_frames

    def tick(self):
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.avg_frames:
            self.timestamps.pop(0)

    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed

    def draw_fps(self, frame):
        fps = self.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame
