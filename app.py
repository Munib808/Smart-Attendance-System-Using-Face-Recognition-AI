import streamlit as st
import cv2
import numpy as np
import os
import time
import shutil
from datetime import datetime

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── imports ───────────────────────────────────────────────────────────────────
from src.detect_face import detect_faces_in_frame, draw_faces_on_frame, crop_face_from_frame
from src.train_embeddings import (
    load_embeddings, add_student_embeddings, rebuild_all_embeddings,
    extract_embedding_from_array, check_face_duplicate, get_dataset_stats
)
from src.attendance_csv import (
    ensure_attendance_file, load_attendance_df, mark_attendance,
    get_all_dates, get_attendance_summary, export_attendance_csv
)
from src.pdf_export import generate_pdf_report
from src.recognize import recognize_faces_in_frame, draw_recognition_results, FPSCounter

# ── directory setup ───────────────────────────────────────────────────────────
RAW_DIR   = os.path.join("dataset", "raw")
EMBED_DIR = os.path.join("dataset", "embeddings")
for d in [RAW_DIR, EMBED_DIR, "attendance", os.path.join("attendance", "reports")]:
    os.makedirs(d, exist_ok=True)

ensure_attendance_file()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    .main-header {
        background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
        color:white; padding:2rem 2.5rem; border-radius:16px; margin-bottom:1.5rem;
        box-shadow:0 8px 32px rgba(15,52,96,.3);
    }
    .main-header h1 { margin:0; font-size:2rem; font-weight:700; letter-spacing:-.5px; }
    .main-header p  { margin:.4rem 0 0; opacity:.75; font-size:.95rem; }
    .metric-card {
        background:white; border-radius:12px; padding:1.2rem 1.5rem;
        border-left:4px solid #0f3460; margin-bottom:.8rem;
        box-shadow:0 2px 12px rgba(0,0,0,.06);
    }
    .metric-card .label { font-size:.8rem; color:#888; font-weight:600; text-transform:uppercase; letter-spacing:.5px; }
    .metric-card .value { font-size:2rem; font-weight:700; color:#1a1a2e; }
    .status-badge { display:inline-block; padding:.25rem .8rem; border-radius:20px; font-size:.8rem; font-weight:600; }
    .badge-present { background:#d4edda; color:#155724; }
    .badge-already { background:#fff3cd; color:#856404; }
    .badge-unknown { background:#f8d7da; color:#721c24; }
    .info-box    { background:#e8f0fe; border-left:4px solid #4285f4; border-radius:8px; padding:.8rem 1rem; margin:.5rem 0; font-size:.9rem; color:#1a1a2e; }
    .warn-box    { background:#fff3cd; border-left:4px solid #ffc107; border-radius:8px; padding:.8rem 1rem; margin:.5rem 0; font-size:.9rem; color:#856404; }
    .success-box { background:#d4edda; border-left:4px solid #28a745; border-radius:8px; padding:.8rem 1rem; margin:.5rem 0; font-size:.9rem; color:#155724; }
    .error-box   { background:#f8d7da; border-left:4px solid #dc3545; border-radius:8px; padding:.8rem 1rem; margin:.5rem 0; font-size:.9rem; color:#721c24; }
    div[data-testid="stSidebar"] { background:#1a1a2e; }
    div[data-testid="stSidebar"] * { color:#e0e0e0 !important; }
    div[data-testid="stSidebar"] .stSelectbox label { color:#aab4d0 !important; }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Smart Attendance System</h1>
    <p>Real-time face recognition attendance powered by DeepFace FaceNet</p>
</div>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Navigation")
    page = st.selectbox(
        "Select Module",
        ["🏠 Dashboard", "👤 Register Student", "🧠 Train Embeddings",
         "📸 Take Attendance", "📊 View Attendance", "📥 Export Reports",
         "🗑️ Manage Students", "🖼️ View Registered Images"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    stats = get_dataset_stats()
    st.markdown(f"**Students Registered:** {stats['total_students']}")
    st.markdown(f"**Total Images:** {stats['total_images']}")
    st.markdown(f"**Embeddings Trained:** {stats['students_with_embeddings']}")
    att_summary = get_attendance_summary()
    st.markdown(f"**Today's Attendance:** {att_summary['today_count']}")
    st.markdown("---")
    st.markdown("*DeepFace · FaceNet · OpenCV*")

# =============================================================================
# DASHBOARD
# =============================================================================
if page == "🏠 Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    for col, label, val in [
        (col1, "Students",           stats['total_students']),
        (col2, "Training Images",    stats['total_images']),
        (col3, "Embeddings",         stats['total_embeddings']),
        (col4, "Today's Attendance", att_summary['today_count']),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                        f'<div class="value">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("### 📅 Today's Attendance")
    today = datetime.now().strftime("%Y-%m-%d")
    df = load_attendance_df()
    if not df.empty and "Date" in df.columns:
        today_df = df[df["Date"] == today]
        if not today_df.empty:
            st.dataframe(today_df, use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="info-box">No attendance marked today yet.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">No attendance records found.</div>', unsafe_allow_html=True)

    st.markdown("### 🗂️ Registered Students")
    if stats["students"]:
        cols = st.columns(3)
        for i, s in enumerate(stats["students"]):
            with cols[i % 3]:
                st.markdown(f'<div class="metric-card"><div class="label">{s["name"]}</div>'
                            f'<div class="value" style="font-size:1.2rem">{s["images"]} images</div></div>',
                            unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">No students registered yet.</div>', unsafe_allow_html=True)

# =============================================================================
# REGISTER STUDENT
# =============================================================================
elif page == "👤 Register Student":
    st.markdown("## 👤 Student Registration")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("### Student Details")
        student_name = st.text_input("Student Name", placeholder="Enter full name...", key="reg_name")
        num_images   = st.slider("Images to Capture", 20, 50, 35, key="reg_num")

        cancel_flag = False
        if student_name and student_name.strip():
            sf = os.path.join(RAW_DIR, student_name.strip())
            if os.path.exists(sf):
                existing_imgs = len([f for f in os.listdir(sf)
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                st.markdown(f'<div class="warn-box">⚠️ Student "<b>{student_name.strip()}</b>" already exists with {existing_imgs} images.</div>',
                            unsafe_allow_html=True)
                action = st.radio("Action", ["Add More Photos", "Cancel"], key="reg_action")
                if action == "Cancel":
                    cancel_flag = True

        start_capture = st.button("🎥 Start Capture", type="primary", key="start_cap",
                                   disabled=(not bool(student_name and student_name.strip())) or cancel_flag)

    with col_right:
        st.markdown("### Live Camera Feed")
        cam_ph    = st.empty()
        prog_ph   = st.empty()
        status_ph = st.empty()

    if start_capture and student_name and student_name.strip() and not cancel_flag:
        sname   = student_name.strip()
        sfolder = os.path.join(RAW_DIR, sname)
        os.makedirs(sfolder, exist_ok=True)

        # Duplicate-face check for new students
        existing_embeddings = load_embeddings()
        if existing_embeddings and sname not in existing_embeddings:
            status_ph.markdown('<div class="info-box">🔍 Checking for duplicate faces...</div>', unsafe_allow_html=True)
            cap_chk   = cv2.VideoCapture(0)
            probe_emb = None
            if cap_chk.isOpened():
                for _ in range(30):
                    ret, frm = cap_chk.read()
                    if ret and frm is not None:
                        fcs = detect_faces_in_frame(frm)
                        if fcs:
                            cr = crop_face_from_frame(frm, fcs[0], padding=15)
                            if cr is not None and cr.size > 0:
                                try:
                                    probe_emb = extract_embedding_from_array(cv2.resize(cr, (160, 160)))
                                    if probe_emb is not None:
                                        break
                                except Exception:
                                    pass
                cap_chk.release()
            if probe_emb is not None:
                is_dup, mname, sim = check_face_duplicate(probe_emb, existing_embeddings)
                if is_dup:
                    st.markdown(f'<div class="error-box">🚫 Face already registered as <b>{mname}</b> ({sim:.2%}). Blocked.</div>',
                                unsafe_allow_html=True)
                    st.stop()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open webcam.")
            st.stop()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        captured       = 0
        last_save_time = time.time()
        save_interval  = 0.3

        stop_btn = st.button("⏹ Stop Capture", key="stop_cap")
        status_ph.markdown('<div class="info-box">📸 Capturing — please face the camera.</div>', unsafe_allow_html=True)

        while captured < num_images and not stop_btn:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            display = frame.copy()
            now     = time.time()
            faces   = detect_faces_in_frame(frame)

            if faces:
                fbox      = faces[0]
                face_crop = crop_face_from_frame(frame, fbox, padding=20)

                if face_crop is not None and face_crop.size > 0:
                    if (now - last_save_time) >= save_interval:
                        img_path = os.path.join(sfolder, f"{sname}_{captured + 1:03d}.jpg")
                        cv2.imwrite(img_path, face_crop)
                        captured      += 1
                        last_save_time = now

                display = draw_faces_on_frame(frame, [fbox],
                                              labels=[f"Capturing {captured}/{num_images}"],
                                              colors=[(0, 255, 0)])
            else:
                cv2.putText(display, "No face detected", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cam_ph.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            prog_ph.progress(captured / num_images, text=f"Captured {captured}/{num_images} images")

        cap.release()

        if captured > 0:
            status_ph.markdown(
                f'<div class="success-box">✅ Captured {captured} images for <b>{sname}</b>. Generating embeddings...</div>',
                unsafe_allow_html=True)
            with st.spinner("🧬 Generating FaceNet embeddings..."):
                count = add_student_embeddings(sname)
            if count > 0:
                st.markdown(f'<div class="success-box">🎉 Done! {count} embeddings for <b>{sname}</b>.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn-box">⚠️ No embeddings generated. Check image quality.</div>',
                            unsafe_allow_html=True)
        else:
            status_ph.markdown('<div class="warn-box">⚠️ No images captured. Ensure face is visible.</div>',
                               unsafe_allow_html=True)

# =============================================================================
# TRAIN EMBEDDINGS
# =============================================================================
elif page == "🧠 Train Embeddings":
    st.markdown("## 🧠 Embedding Management")
    stats = get_dataset_stats()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="label">Students in Dataset</div>'
                    f'<div class="value">{stats["total_students"]}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="label">Students with Embeddings</div>'
                    f'<div class="value">{stats["students_with_embeddings"]}</div></div>', unsafe_allow_html=True)

    st.markdown("### 👤 Generate Embeddings for Single Student")
    if stats["students"]:
        sel = st.selectbox("Select Student", [s["name"] for s in stats["students"]], key="train_sel")
        if st.button("🧬 Generate Embeddings", type="primary", key="train_go"):
            prog = st.progress(0)
            def cb(d, t): prog.progress(d / t, text=f"Processing {d}/{t}")
            with st.spinner(f"Generating for {sel}..."):
                count = add_student_embeddings(sel, cb)
            if count > 0:
                st.markdown(f'<div class="success-box">✅ {count} embeddings for <b>{sel}</b>.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box">❌ No embeddings generated.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">No students found.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔄 Full Rebuild (Admin)")
    st.markdown('<div class="warn-box">⚠️ Rebuilds ALL embeddings from scratch.</div>', unsafe_allow_html=True)
    if st.checkbox("I understand this will re-process all students", key="rebuild_ack"):
        if st.button("🔄 Rebuild All", type="secondary", key="rebuild_go"):
            prog    = st.progress(0)
            stat_ph = st.empty()
            def rcb(d, t, n):
                prog.progress(d / t, text=f"{n} ({d}/{t})")
                stat_ph.markdown(f'<div class="info-box">Processing <b>{n}</b></div>', unsafe_allow_html=True)
            with st.spinner("Rebuilding..."):
                result = rebuild_all_embeddings(rcb)
            total = sum(len(v) for v in result.values())
            st.markdown(f'<div class="success-box">✅ {len(result)} students — {total} embeddings.</div>',
                        unsafe_allow_html=True)

# =============================================================================
# TAKE ATTENDANCE
# =============================================================================
elif page == "📸 Take Attendance":
    st.markdown("## 📸 Real-Time Attendance")

    embeddings = load_embeddings()
    if not embeddings:
        st.markdown('<div class="error-box">❌ No embeddings. Register students first.</div>', unsafe_allow_html=True)
        st.stop()

    col_cam, col_stat = st.columns([2, 1])
    with col_cam:
        st.markdown("### 📹 Live Feed")
        cam_ph = st.empty()
        fps_ph = st.empty()
    with col_stat:
        st.markdown("### 📋 Attendance Status")
        stat_ph  = st.empty()
        today_ph = st.empty()

    if "attendance_session" not in st.session_state:
        st.session_state.attendance_session = {}
    if "att_running" not in st.session_state:
        st.session_state.att_running = False

    b1, b2 = st.columns(2)
    with b1:
        start_btn = st.button("▶️ Start Attendance", type="primary",  key="att_start",
                               disabled=st.session_state.att_running)
    with b2:
        stop_btn  = st.button("⏹ Stop",              type="secondary", key="att_stop",
                               disabled=not st.session_state.att_running)

    if start_btn and not st.session_state.att_running:
        st.session_state.att_running = True
        st.session_state.attendance_session = {}
        st.rerun()

    if stop_btn and st.session_state.att_running:
        st.session_state.att_running = False
        st.rerun()

    if st.session_state.att_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open webcam.")
            st.session_state.att_running = False
            st.stop()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fps_counter       = FPSCounter()
        frame_count       = 0
        attendance_status = {}

        while st.session_state.att_running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            fps_counter.tick()
            annotated = frame.copy()

            try:
                # Run recognition every 3rd frame
                if frame_count % 3 == 0:
                    results = recognize_faces_in_frame(frame, embeddings)
                    for res in results:
                        name = res["name"]
                        if name != "Unknown":
                            r = mark_attendance(name)
                            if r == "marked":
                                attendance_status[name] = "Present"
                                st.session_state.attendance_session[name] = "✅ Present"
                            elif r == "already_marked":
                                attendance_status[name] = "Already Marked"
                                st.session_state.attendance_session[name] = "🟡 Already Marked"
                        else:
                            attendance_status["Unknown"] = "Unknown"
                    annotated = draw_recognition_results(frame, results, attendance_status)
            except Exception:
                pass

            fps_counter.draw_fps(annotated)
            cam_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            fps_ph.markdown(f"**FPS:** {fps_counter.get_fps():.1f}")

            if st.session_state.attendance_session:
                html = ""
                for name, status in st.session_state.attendance_session.items():
                    bc = "badge-present" if "Present" in status else "badge-already"
                    html += (f'<div style="padding:.4rem 0;border-bottom:1px solid #eee;">'
                             f'<b>{name}</b><br><span class="status-badge {bc}">{status}</span></div>')
                stat_ph.markdown(html, unsafe_allow_html=True)

            today = datetime.now().strftime("%Y-%m-%d")
            df = load_attendance_df()
            if not df.empty and "Date" in df.columns:
                td = df[df["Date"] == today]
                if not td.empty:
                    today_ph.dataframe(td[["Name", "Time", "Status"]], use_container_width=True, hide_index=True)

            if not st.session_state.get("att_running", False):
                break

        cap.release()
        n = len(st.session_state.attendance_session)
        st.markdown(f'<div class="success-box">✅ Session ended. Marked {n} student(s).</div>',
                    unsafe_allow_html=True)

# =============================================================================
# VIEW ATTENDANCE
# =============================================================================
elif page == "📊 View Attendance":
    st.markdown("## 📊 Attendance Records")
    df = load_attendance_df()
    if df.empty:
        st.markdown('<div class="info-box">No attendance records yet.</div>', unsafe_allow_html=True)
    else:
        summary = get_attendance_summary()
        c1, c2, c3 = st.columns(3)
        for col, lbl, val in [
            (c1, "Total Records",   summary['total_records']),
            (c2, "Unique Students", summary['unique_students']),
            (c3, "Days Recorded",   summary['unique_dates']),
        ]:
            with col:
                st.markdown(f'<div class="metric-card"><div class="label">{lbl}</div>'
                            f'<div class="value">{val}</div></div>', unsafe_allow_html=True)

        st.markdown("### 🔍 Filter Records")
        cf1, cf2 = st.columns(2)
        with cf1:
            sel_date = st.selectbox("Filter by Date", ["All Dates"] + get_all_dates(), key="vw_date")
        with cf2:
            srch = st.text_input("Search by Name", placeholder="Leave empty for all...", key="vw_name")

        fdf = df.copy()
        if sel_date != "All Dates":
            fdf = fdf[fdf["Date"] == sel_date]
        if srch.strip():
            fdf = fdf[fdf["Name"].str.contains(srch.strip(), case=False, na=False)]
        st.markdown(f"**Showing {len(fdf)} records**")
        st.dataframe(fdf, use_container_width=True, hide_index=True)

# =============================================================================
# EXPORT REPORTS
# =============================================================================
elif page == "📥 Export Reports":
    st.markdown("## 📥 Export Reports")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📋 CSV Export")
        csv_date = st.selectbox("Select Date for CSV", ["All Records"] + get_all_dates(), key="csv_date")
        if st.button("📥 Prepare CSV", type="primary", key="csv_btn"):
            csv_path = export_attendance_csv() if csv_date == "All Records" else export_attendance_csv(csv_date)
            if os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    st.download_button("⬇️ Download CSV", f.read(),
                                       file_name=os.path.basename(csv_path),
                                       mime="text/csv", key="csv_dl")
    with c2:
        st.markdown("### 📄 PDF Report")
        pdf_date = st.selectbox("Select Date for PDF", ["Today"] + get_all_dates(), key="pdf_date")
        if st.button("📄 Generate PDF", type="primary", key="pdf_btn"):
            dstr = datetime.now().strftime("%Y-%m-%d") if pdf_date == "Today" else pdf_date
            with st.spinner("Generating..."):
                try:
                    pdf_path = generate_pdf_report(date_str=dstr)
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f:
                            st.download_button("⬇️ Download PDF", f.read(),
                                               file_name=os.path.basename(pdf_path),
                                               mime="application/pdf", key="pdf_dl")
                        st.markdown(f'<div class="success-box">✅ PDF: {pdf_path}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="error-box">❌ PDF generation failed.</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-box">❌ {e}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📁 Existing Reports")
    rdir = os.path.join("attendance", "reports")
    if os.path.exists(rdir):
        reports = sorted([f for f in os.listdir(rdir) if f.endswith(".pdf")], reverse=True)
        if reports:
            for r in reports:
                with open(os.path.join(rdir, r), "rb") as f:
                    st.download_button(f"📄 {r}", f.read(), file_name=r, mime="application/pdf", key=f"rpt_{r}")
        else:
            st.markdown('<div class="info-box">No PDF reports yet.</div>', unsafe_allow_html=True)

# =============================================================================
# MANAGE STUDENTS
# =============================================================================
elif page == "🗑️ Manage Students":
    st.markdown("## 🗑️ Manage / Remove Students")
    st.markdown('<div class="warn-box">⚠️ Removing a student permanently deletes their images, embeddings, and attendance records.</div>',
                unsafe_allow_html=True)

    stats = get_dataset_stats()
    if not stats["students"]:
        st.markdown('<div class="info-box">No students registered.</div>', unsafe_allow_html=True)
        st.stop()

    sel = st.selectbox("Select Student to Remove", [s["name"] for s in stats["students"]], key="mgmt_sel")

    if sel:
        sfolder = os.path.join(RAW_DIR, sel)
        img_cnt = len([f for f in os.listdir(sfolder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(sfolder) else 0
        embs    = load_embeddings()
        emb_cnt = len(embs.get(sel, []))

        st.markdown(f'<div class="metric-card"><div class="label">{sel}</div>'
                    f'<div class="value" style="font-size:1rem">📷 {img_cnt} images &nbsp;|&nbsp; 🧬 {emb_cnt} embeddings</div></div>',
                    unsafe_allow_html=True)

        confirm = st.checkbox(f'Confirm: permanently remove "{sel}"', key="mgmt_confirm")
        if confirm:
            if st.button("🗑️ Remove Student", type="primary", key="mgmt_go"):
                parts = []
                if os.path.exists(sfolder):
                    shutil.rmtree(sfolder)
                    parts.append(f"{img_cnt} images")
                if sel in embs:
                    del embs[sel]
                    from src.train_embeddings import save_embeddings
                    save_embeddings(embs)
                    parts.append(f"{emb_cnt} embeddings")
                att_df = load_attendance_df()
                if not att_df.empty and "Name" in att_df.columns:
                    n_att  = att_df[att_df["Name"] == sel].shape[0]
                    att_df = att_df[att_df["Name"] != sel]
                    from src.attendance_csv import ATTENDANCE_FILE
                    att_df.to_csv(ATTENDANCE_FILE, index=False)
                    if n_att:
                        parts.append(f"{n_att} attendance records")
                st.markdown(f'<div class="success-box">✅ <b>{sel}</b> removed ({", ".join(parts) or "nothing on disk"}).</div>',
                            unsafe_allow_html=True)
                st.rerun()

    st.markdown("---")
    st.markdown("### 👥 All Students")
    for s in stats["students"]:
        st.markdown(f"- **{s['name']}** — {s['images']} images")

# =============================================================================
# VIEW REGISTERED IMAGES
# =============================================================================
elif page == "🖼️ View Registered Images":
    st.markdown("## 🖼️ Registered Student Images")
    stats = get_dataset_stats()
    if not stats["students"]:
        st.markdown('<div class="info-box">No students registered.</div>', unsafe_allow_html=True)
        st.stop()

    sel = st.selectbox("Select Student", [s["name"] for s in stats["students"]], key="imgv_sel")
    if sel:
        sfolder = os.path.join(RAW_DIR, sel)
        if not os.path.exists(sfolder):
            st.markdown('<div class="error-box">Image folder not found.</div>', unsafe_allow_html=True)
            st.stop()

        imgs = sorted([f for f in os.listdir(sfolder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not imgs:
            st.markdown('<div class="info-box">No images for this student.</div>', unsafe_allow_html=True)
            st.stop()

        st.markdown(f"### 📷 {sel} — {len(imgs)} images")
        PER_PAGE    = 30
        total_pages = max(1, (len(imgs) + PER_PAGE - 1) // PER_PAGE)
        pg          = st.number_input("Page", 1, total_pages, 1, key="imgv_pg")
        s0, s1      = (pg - 1) * PER_PAGE, min(pg * PER_PAGE, len(imgs))
        st.markdown(f"*Showing {s0+1}–{s1} of {len(imgs)}*")
        cols = st.columns(5)
        for i, fn in enumerate(imgs[s0:s1]):
            fp = os.path.join(sfolder, fn)
            try:
                img = cv2.imread(fp)
                if img is not None:
                    with cols[i % 5]:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=fn, use_container_width=True)
            except Exception:
                with cols[i % 5]:
                    st.markdown(f"*{fn}*")
