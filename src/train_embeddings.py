import os
import pickle
import numpy as np
from deepface import DeepFace


EMBEDDINGS_PATH = os.path.join("dataset", "embeddings", "embeddings.pkl")
RAW_DATASET_PATH = os.path.join("dataset", "raw")
MODEL_NAME = "Facenet"


def load_embeddings():
    """Load existing embeddings from disk. Returns dict."""
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            with open(EMBEDDINGS_PATH, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_embeddings(embeddings_dict):
    """Save embeddings dict to disk."""
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings_dict, f)


def extract_embedding_from_image(image_path):
    """
    Extract FaceNet embedding from a single image.
    Returns embedding list or None on failure.
    """
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False
        )
        if result and len(result) > 0:
            embedding = result[0].get("embedding", None)
            if embedding is not None:
                return np.array(embedding, dtype=np.float32)
    except Exception:
        pass
    return None


def extract_embedding_from_array(img_array):
    """
    Extract FaceNet embedding from a numpy array.
    Returns embedding array or None on failure.
    """
    try:
        if img_array is None or img_array.size == 0:
            return None
        result = DeepFace.represent(
            img_path=img_array,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False
        )
        if result and len(result) > 0:
            embedding = result[0].get("embedding", None)
            if embedding is not None:
                return np.array(embedding, dtype=np.float32)
    except Exception:
        pass
    return None


def generate_embeddings_for_student(student_name, progress_callback=None):
    """
    Generate embeddings for a single student.
    Returns list of embeddings.
    """
    student_folder = os.path.join(RAW_DATASET_PATH, student_name)
    if not os.path.exists(student_folder):
        return []

    embeddings = []
    image_files = [
        f for f in os.listdir(student_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(student_folder, img_file)
        emb = extract_embedding_from_image(img_path)
        if emb is not None:
            embeddings.append(emb)

        if progress_callback:
            progress_callback(idx + 1, len(image_files))

    return embeddings


def add_student_embeddings(student_name, progress_callback=None):
    """
    Incrementally add embeddings for a new student.
    Loads existing embeddings, appends new ones, saves back.
    Returns number of embeddings generated.
    """
    existing = load_embeddings()
    new_embeddings = generate_embeddings_for_student(student_name, progress_callback)

    if new_embeddings:
        existing[student_name] = new_embeddings
        save_embeddings(existing)

    return len(new_embeddings)


def rebuild_all_embeddings(progress_callback=None):
    """
    Full rebuild of all embeddings. Admin only.
    Returns dict of all embeddings.
    """
    all_embeddings = {}

    if not os.path.exists(RAW_DATASET_PATH):
        return all_embeddings

    students = [
        d for d in os.listdir(RAW_DATASET_PATH)
        if os.path.isdir(os.path.join(RAW_DATASET_PATH, d))
    ]

    for student_idx, student_name in enumerate(students):
        embs = generate_embeddings_for_student(student_name)
        if embs:
            all_embeddings[student_name] = embs

        if progress_callback:
            progress_callback(student_idx + 1, len(students), student_name)

    save_embeddings(all_embeddings)
    return all_embeddings


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def check_face_duplicate(face_embedding, embeddings_dict, threshold=0.70):
    """
    Check if a face embedding matches any stored embedding.
    Returns (is_duplicate, matched_name, max_similarity).
    """
    if face_embedding is None or not embeddings_dict:
        return False, None, 0.0

    best_name = None
    best_sim = 0.0

    for student_name, emb_list in embeddings_dict.items():
        for stored_emb in emb_list:
            sim = cosine_similarity(face_embedding, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = student_name

    if best_sim >= threshold:
        return True, best_name, best_sim

    return False, None, best_sim


def get_dataset_stats():
    """Return stats about the current dataset."""
    stats = {
        "total_students": 0,
        "total_images": 0,
        "students_with_embeddings": 0,
        "total_embeddings": 0,
        "students": []
    }

    if os.path.exists(RAW_DATASET_PATH):
        for student in os.listdir(RAW_DATASET_PATH):
            student_path = os.path.join(RAW_DATASET_PATH, student)
            if os.path.isdir(student_path):
                images = [f for f in os.listdir(student_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                stats["total_students"] += 1
                stats["total_images"] += len(images)
                stats["students"].append({"name": student, "images": len(images)})

    embeddings = load_embeddings()
    stats["students_with_embeddings"] = len(embeddings)
    stats["total_embeddings"] = sum(len(v) for v in embeddings.values())

    return stats
