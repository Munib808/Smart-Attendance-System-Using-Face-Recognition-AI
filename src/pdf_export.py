import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from src.attendance_csv import load_attendance_df, get_attendance_for_date


REPORTS_DIR = os.path.join("attendance", "reports")


def ensure_reports_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def generate_pdf_report(date_str=None, output_path=None):
    """
    Generate a PDF attendance report.
    date_str: filter by date (None = all records)
    output_path: where to save. If None, auto-generate.
    Returns path to the PDF.
    """
    ensure_reports_dir()

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        df = load_attendance_df()
    else:
        df = get_attendance_for_date(date_str)

    if output_path is None:
        output_path = os.path.join(REPORTS_DIR, f"report_{date_str}.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1a1a2e"),
        alignment=TA_CENTER,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.HexColor("#4a4a6a"),
        alignment=TA_CENTER,
        spaceAfter=4
    )

    info_style = ParagraphStyle(
        "InfoStyle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#666666"),
        alignment=TA_CENTER,
        spaceAfter=2
    )

    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#1a1a2e"),
        alignment=TA_LEFT,
        spaceBefore=10,
        spaceAfter=6,
        fontName="Helvetica-Bold"
    )

    story = []

    # Title
    story.append(Paragraph("Smart Attendance System", title_style))
    story.append(Paragraph("Automated Face Recognition Attendance Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 0.3 * inch))

    # Report metadata
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Report Date: {date_str}", info_style))
    story.append(Paragraph(f"Generated At: {generated_at}", info_style))
    story.append(Paragraph(f"Total Records: {len(df)}", info_style))
    story.append(Spacer(1, 0.2 * inch))

    # Summary section
    if not df.empty and "Name" in df.columns:
        unique_students = df["Name"].nunique()
        present_count = len(df[df["Status"] == "Present"]) if "Status" in df.columns else len(df)
        story.append(Paragraph("Summary", section_style))

        summary_data = [
            ["Metric", "Value"],
            ["Total Attendance Records", str(len(df))],
            ["Unique Students Present", str(unique_students)],
            ["Present Count", str(present_count)],
        ]

        summary_table = Table(summary_data, colWidths=[10 * cm, 6 * cm])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f8fc"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccccdd")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3 * inch))

    # Attendance detail table
    story.append(Paragraph("Attendance Details", section_style))

    if df.empty:
        story.append(Paragraph("No attendance records found for the selected date.", info_style))
    else:
        # Build table data
        display_cols = ["Name", "Date", "Time", "Status"]
        available_cols = [c for c in display_cols if c in df.columns]

        table_data = [available_cols]
        for _, row in df[available_cols].iterrows():
            table_data.append([str(v) for v in row.tolist()])

        col_widths_map = {
            "Name": 6 * cm,
            "Date": 4 * cm,
            "Time": 4 * cm,
            "Status": 4 * cm,
        }
        col_widths = [col_widths_map.get(c, 4 * cm) for c in available_cols]

        detail_table = Table(table_data, colWidths=col_widths, repeatRows=1)

        row_styles = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccccdd")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]

        # Alternate row colors
        for i in range(1, len(table_data)):
            bg = colors.HexColor("#f0f0f8") if i % 2 == 0 else colors.white
            row_styles.append(("BACKGROUND", (0, i), (-1, i), bg))

            # Color "Present" green, others orange
            if "Status" in available_cols:
                status_col_idx = available_cols.index("Status")
                status_val = table_data[i][status_col_idx] if status_col_idx < len(table_data[i]) else ""
                if status_val == "Present":
                    row_styles.append(("TEXTCOLOR", (status_col_idx, i), (status_col_idx, i), colors.HexColor("#2d6a4f")))
                    row_styles.append(("FONTNAME", (status_col_idx, i), (status_col_idx, i), "Helvetica-Bold"))

        detail_table.setStyle(TableStyle(row_styles))
        story.append(detail_table)

    story.append(Spacer(1, 0.5 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#ccccdd")))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "Generated by Smart Attendance System | Powered by DeepFace FaceNet",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor("#aaaaaa"), alignment=TA_CENTER)
    ))

    doc.build(story)
    return output_path
