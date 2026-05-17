# 🎓 Smart Attendance System

A production-ready, real-time face recognition attendance system built with **DeepFace FaceNet**, **OpenCV**, and **Streamlit**.

---

## 📁 Project Structure

```
Smart-Attendance/
├── app.py                    # Main Streamlit application
├── requirements.txt
├── README.md
├── dataset/
│   ├── raw/                  # Student face images: dataset/raw/<name>/*.jpg
│   └── embeddings/
│       └── embeddings.pkl    # Stored FaceNet embeddings
├── attendance/
│   ├── attendance.csv        # Main attendance log
│   └── reports/              # Generated PDF reports
├── src/
│   ├── __init__.py
│   ├── capture.py            # Webcam capture utilities
│   ├── detect_face.py        # OpenCV face detection
│   ├── recognize.py          # Face recognition + FPS counter
│   ├── train_embeddings.py   # FaceNet embedding engine
│   ├── attendance_csv.py     # CSV-based attendance storage
│   └── pdf_export.py         # ReportLab PDF generation
└── assets/
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Smart-Attendance

# 2. Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

---

## 🚀 Usage Workflow

### Step 1 — Register a Student
1. Navigate to **Register Student** in the sidebar
2. Enter the student's name
3. Click **Start Capture** — the system will:
   - Check if the name already exists
   - Perform a **face duplicate check** (compares against all stored faces)
   - Automatically capture 35 face images from your webcam
   - Generate FaceNet embeddings and save them incrementally

### Step 2 — Train Embeddings
- Embeddings are auto-generated after each registration
- For manual control, use the **Train Embeddings** page
- Use **Full Rebuild** only if the embeddings file becomes corrupted

### Step 3 — Take Attendance
1. Navigate to **Take Attendance**
2. Click **Start Attendance**
3. The system will:
   - Detect all faces in the frame simultaneously (**multi-face support**)
   - Recognize each face against stored embeddings
   - Mark attendance in `attendance.csv` (once per student per day)
   - Display status: ✅ Present | 🟡 Already Marked | ❓ Unknown
4. Click **Stop** to end the session

### Step 4 — View & Export
- **View Attendance**: Filter by date or student name
- **Export Reports**: Download CSV or generate a PDF report

---

## 🧬 Incremental Embedding System

The system **never retrains existing students**. When a new student is registered:

1. Images are saved to `dataset/raw/<student_name>/`
2. `add_student_embeddings()` is called for ONLY that student
3. The existing `embeddings.pkl` is loaded
4. New embeddings are appended under the student's key
5. The file is saved back

Embedding format:
```python
{
    "Ali":   [array(128,), array(128,), ...],
    "Ahmed": [array(128,), array(128,), ...],
}
```

---

## 🎯 Technical Details

| Component | Technology |
|-----------|-----------|
| Face Detection | OpenCV Haar Cascade |
| Face Recognition | DeepFace FaceNet (128-dim embeddings) |
| Similarity Metric | Cosine Similarity |
| Recognition Threshold | 0.65 |
| Duplicate Detection Threshold | 0.70 |
| Storage | CSV (attendance) + Pickle (embeddings) |
| UI | Streamlit |
| PDF Reports | ReportLab |

---

## 🛡️ Stability Features

- All DeepFace calls use `detector_backend="skip"` (prevents runtime crashes)
- Every DeepFace call is wrapped in `try/except`
- System **never crashes** on bad input, blurry frames, or missing faces
- Frame processing runs every 3rd frame for smooth real-time performance
- Webcam failures are handled gracefully

---

## 📋 Attendance CSV Format

```
Name,Date,Time,Status
Ali,2024-01-15,09:32:11,Present
Ahmed,2024-01-15,09:33:05,Present
```

- **Unique constraint**: (Name + Date) — no duplicate entries per day
- CSV is auto-created if missing
- All times are local system time

---

## 🔧 Troubleshooting

**Webcam not detected**: Ensure no other application is using the camera. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.

**Low recognition accuracy**: Register more images (40–50), ensure good lighting, and face the camera directly during registration.

**Embeddings not generating**: Check that images in `dataset/raw/<name>/` are clear face photos. Use the Full Rebuild option in Train Embeddings.

**DeepFace model download**: On first run, FaceNet model weights (~90 MB) will be downloaded automatically from GitHub. Ensure internet connectivity.
