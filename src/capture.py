import cv2
import os
import time
from src.detect_face import detect_faces_in_frame


def capture_student_images(student_name, save_dir, num_images=35, progress_callback=None):
    """
    Capture face images for a student using webcam.
    Returns list of saved image paths.
    """
    student_folder = os.path.join(save_dir, student_name)
    os.makedirs(student_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Please check your camera connection.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    saved_paths = []
    captured = 0
    frame_count = 0
    last_capture_time = 0
    capture_interval = 0.3  # seconds between captures

    try:
        while captured < num_images:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_count += 1
            current_time = time.time()

            # Detect faces every 2 frames
            if frame_count % 2 == 0 and (current_time - last_capture_time) >= capture_interval:
                faces = detect_faces_in_frame(frame)

                if faces:
                    # Use first detected face
                    x, y, w, h = faces[0]
                    # Add padding
                    pad = 20
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(frame.shape[1], x + w + pad)
                    y2 = min(frame.shape[0], y + h + pad)

                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        img_path = os.path.join(student_folder, f"{student_name}_{captured + 1:03d}.jpg")
                        cv2.imwrite(img_path, face_crop)
                        saved_paths.append(img_path)
                        captured += 1
                        last_capture_time = current_time

                        if progress_callback:
                            progress_callback(captured, num_images, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return saved_paths


def capture_single_frame():
    """Capture a single frame from webcam. Returns (ret, frame)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read()
    cap.release()
    return ret, frame


def get_webcam_frame(cap):
    """Read a single frame from an open VideoCapture object."""
    if cap is None or not cap.isOpened():
        return False, None
    ret, frame = cap.read()
    return ret, frame


def open_webcam():
    """Open webcam and return VideoCapture object."""
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def release_webcam(cap):
    """Safely release webcam."""
    if cap is not None and cap.isOpened():
        cap.release()
