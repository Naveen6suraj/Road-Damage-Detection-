"""
ROADGUARD AI - Report Generator Module
Module: core/report_generator.py

Generates professional, downloadable road damage inspection reports
in both PDF (using ReportLab) and CSV formats.
"""

import os
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

import config


def generate_csv_report(
    analysis_data: Dict[str, Any],
    detections: List[Dict[str, Any]],
    output_path: str
) -> str:
    """Generate a structured CSV report containing full defect inspection records."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    analysis_id = analysis_data.get("id", "N/A")
    media_type = analysis_data.get("media_type", "image")
    filename = analysis_data.get("filename", "unknown")
    created_at = analysis_data.get("created_at", datetime.now().isoformat())

    fieldnames = [
        "Analysis_ID", "Created_At", "Media_Type", "File_Name",
        "Defect_Number", "Track_ID", "Status", "Class_Code", "Class_Name",
        "Confidence_Percent", "Severity_Score", "Severity_Level",
        "Raw_Severity_Score", "Raw_Severity_Level",
        "Adjusted_Severity_Score", "Adjusted_Severity_Level",
        "Frame_Index", "Timestamp_Sec", "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2",
        "Assessment_Reason"
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        if not detections:
            writer.writerow({
                "Analysis_ID": analysis_id,
                "Created_At": created_at,
                "Media_Type": media_type,
                "File_Name": filename,
                "Defect_Number": 0,
                "Track_ID": "N/A",
                "Status": "N/A",
                "Class_Code": "NONE",
                "Class_Name": "No Defects Detected",
                "Confidence_Percent": "0.0",
                "Severity_Score": "0.0",
                "Severity_Level": "NONE",
                "Raw_Severity_Score": "0.0",
                "Raw_Severity_Level": "NONE",
                "Adjusted_Severity_Score": "0.0",
                "Adjusted_Severity_Level": "NONE",
                "Frame_Index": "N/A",
                "Timestamp_Sec": "N/A",
                "BBox_X1": "0", "BBox_Y1": "0", "BBox_X2": "0", "BBox_Y2": "0",
                "Assessment_Reason": "No road damage detected above confidence threshold."
            })
        else:
            for idx, det in enumerate(detections, 1):
                box = det.get("box", [0, 0, 0, 0])
                if not box and ("box_x1" in det):
                    box = [det["box_x1"], det["box_y1"], det["box_x2"], det["box_y2"]]

                conf = float(det.get("confidence", 0.0)) * 100.0
                is_conf = det.get("is_confirmed", True)
                status_str = "Confirmed" if is_conf else "Transient"
                raw_score = float(det.get("raw_severity_score", det.get("severity_score", 0.0)))
                raw_level = det.get("raw_severity_level", config.get_severity_level(raw_score))
                adj_score = float(det.get("severity_score", 0.0))
                adj_level = det.get("severity_level", config.get_severity_level(adj_score))

                writer.writerow({
                    "Analysis_ID": analysis_id,
                    "Created_At": created_at,
                    "Media_Type": media_type,
                    "File_Name": filename,
                    "Defect_Number": idx,
                    "Track_ID": det.get("track_id", "N/A"),
                    "Status": status_str,
                    "Class_Code": det.get("class_code", "OTHER"),
                    "Class_Name": det.get("class_name", "Unknown"),
                    "Confidence_Percent": f"{conf:.1f}",
                    "Severity_Score": f"{adj_score:.1f}",
                    "Severity_Level": adj_level,
                    "Raw_Severity_Score": f"{raw_score:.1f}",
                    "Raw_Severity_Level": raw_level,
                    "Adjusted_Severity_Score": f"{adj_score:.1f}",
                    "Adjusted_Severity_Level": adj_level,
                    "Frame_Index": det.get("frame_index", "N/A"),
                    "Timestamp_Sec": det.get("timestamp_sec", det.get("first_timestamp_sec", "N/A")),
                    "BBox_X1": round(box[0], 1) if box else 0,
                    "BBox_Y1": round(box[1], 1) if box else 0,
                    "BBox_X2": round(box[2], 1) if box else 0,
                    "BBox_Y2": round(box[3], 1) if box else 0,
                    "Assessment_Reason": det.get("reason", "")
                })

    return output_path


def generate_pdf_report(
    analysis_data: Dict[str, Any],
    detections: List[Dict[str, Any]],
    output_path: str
) -> str:
    """Generate an executive PDF report using ReportLab."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a")
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0284c7")
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6
    )
    cell_style = ParagraphStyle(
        "CellNormal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#334155")
    )
    cell_bold = ParagraphStyle(
        "CellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a")
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#64748b")
    )

    story = []

    # Title Banner
    story.append(Paragraph("ROADGUARD AI | Condition & Damage Report", title_style))
    story.append(Paragraph("Automated Computer Vision Pothole & Road Crack Severity Assessment", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=12))

    # Meta Overview Grid
    created_at = analysis_data.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    meta_data = [
        [
            Paragraph("<b>Analysis ID:</b>", cell_style),
            Paragraph(str(analysis_data.get("id", "N/A")), cell_style),
            Paragraph("<b>Inspection Date:</b>", cell_style),
            Paragraph(created_at[:19].replace("T", " "), cell_style)
        ],
        [
            Paragraph("<b>Source File:</b>", cell_style),
            Paragraph(str(analysis_data.get("filename", "N/A")), cell_style),
            Paragraph("<b>Media Type:</b>", cell_style),
            Paragraph(str(analysis_data.get("media_type", "image")).upper(), cell_style)
        ],
        [
            Paragraph("<b>Total Defects:</b>", cell_style),
            Paragraph(str(analysis_data.get("unique_defects", analysis_data.get("total_defects", 0))), cell_bold),
            Paragraph("<b>Peak Severity:</b>", cell_style),
            Paragraph(str(analysis_data.get("max_severity_level", "LOW")), cell_bold)
        ],
        [
            Paragraph("<b>Average Confidence:</b>", cell_style),
            Paragraph(f"{float(analysis_data.get('avg_confidence', 0.0))*100:.1f}%", cell_style),
            Paragraph("<b>Processing Duration:</b>", cell_style),
            Paragraph(f"{float(analysis_data.get('processing_time_sec', 0.0)):.2f}s", cell_style)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[110, 160, 110, 160])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Severity & Category Executive Breakdown
    story.append(Paragraph("Severity & Defect Class Summary", section_heading))
    
    crit_c = analysis_data.get("critical_count", 0)
    high_c = analysis_data.get("high_count", 0)
    mod_c = analysis_data.get("moderate_count", 0)
    low_c = analysis_data.get("low_count", 0)

    # Class distribution
    meta_dict = analysis_data.get("metadata", {})
    class_dist = meta_dict.get("damage_type_distribution", {})

    summary_data = [
        [
            Paragraph("<b>Severity Breakdown</b>", cell_bold),
            Paragraph("<b>Count</b>", cell_bold),
            Paragraph("<b>Damage Category</b>", cell_bold),
            Paragraph("<b>Code</b>", cell_bold),
            Paragraph("<b>Count</b>", cell_bold)
        ],
        [
            Paragraph("CRITICAL (Score 80-100)", cell_style),
            Paragraph(str(crit_c), cell_bold),
            Paragraph("Pothole", cell_style),
            Paragraph("D40", cell_style),
            Paragraph(str(class_dist.get("D40", 0)), cell_style)
        ],
        [
            Paragraph("HIGH (Score 60-79)", cell_style),
            Paragraph(str(high_c), cell_bold),
            Paragraph("Alligator Crack", cell_style),
            Paragraph("D20", cell_style),
            Paragraph(str(class_dist.get("D20", 0)), cell_style)
        ],
        [
            Paragraph("MODERATE (Score 30-59)", cell_style),
            Paragraph(str(mod_c), cell_bold),
            Paragraph("Transverse Crack", cell_style),
            Paragraph("D10", cell_style),
            Paragraph(str(class_dist.get("D10", 0)), cell_style)
        ],
        [
            Paragraph("LOW (Score 0-29)", cell_style),
            Paragraph(str(low_c), cell_bold),
            Paragraph("Longitudinal Crack", cell_style),
            Paragraph("D00", cell_style),
            Paragraph(str(class_dist.get("D00", 0)), cell_style)
        ],
        [
            Paragraph("<b>Total Unique Defects</b>", cell_bold),
            Paragraph(f"<b>{crit_c + high_c + mod_c + low_c}</b>", cell_bold),
            Paragraph("Other Corruption", cell_style),
            Paragraph("OTHER", cell_style),
            Paragraph(str(class_dist.get("OTHER", 0)), cell_style)
        ]
    ]

    summary_table = Table(summary_data, colWidths=[150, 60, 160, 60, 110])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TEXTCOLOR", (0, 1), (0, 1), colors.HexColor("#dc2626")),
        ("TEXTCOLOR", (0, 2), (0, 2), colors.HexColor("#ea580c")),
        ("TEXTCOLOR", (0, 3), (0, 3), colors.HexColor("#d97706")),
        ("TEXTCOLOR", (0, 4), (0, 4), colors.HexColor("#16a34a")),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # Defect Registry Table
    story.append(Paragraph("Detailed Defect Registry", section_heading))

    table_rows = [
        [
            Paragraph("<b>#</b>", cell_bold),
            Paragraph("<b>Track ID</b>", cell_bold),
            Paragraph("<b>Category (Code)</b>", cell_bold),
            Paragraph("<b>Status</b>", cell_bold),
            Paragraph("<b>Conf</b>", cell_bold),
            Paragraph("<b>Score (Raw/Adj)</b>", cell_bold),
            Paragraph("<b>Severity</b>", cell_bold),
            Paragraph("<b>Assessment Rationale</b>", cell_bold)
        ]
    ]

    if not detections:
        table_rows.append([
            Paragraph("1", cell_style),
            Paragraph("-", cell_style),
            Paragraph("No Defects Detected", cell_style),
            Paragraph("-", cell_style),
            Paragraph("-", cell_style),
            Paragraph("0.0 / 0.0", cell_style),
            Paragraph("NONE", cell_style),
            Paragraph("No pavement defects observed above the specified confidence threshold.", cell_style)
        ])
    else:
        # Cap printed rows in PDF to 50 for clean presentation
        for idx, det in enumerate(detections[:50], 1):
            is_conf = det.get("is_confirmed", True)
            status_text = "Confirmed" if is_conf else "Transient"
            raw_s = float(det.get("raw_severity_score", det.get("severity_score", 0.0)))
            adj_s = float(det.get("severity_score", 0.0))
            sev_lvl = det.get("severity_level", config.get_severity_level(adj_s))
            conf = float(det.get("confidence", 0.0)) * 100
            code = det.get("class_code", "OTHER")
            name = det.get("class_name", "Unknown")
            track_id = str(det.get("track_id", "-"))
            reason = det.get("reason", "")
            # Truncate reason if too long
            if len(reason) > 95:
                reason = reason[:92] + "..."

            status_style = cell_bold if is_conf else cell_style
            score_text = f"{raw_s:.1f} / {adj_s:.1f}"

            table_rows.append([
                Paragraph(str(idx), cell_style),
                Paragraph(track_id, cell_style),
                Paragraph(f"{name} ({code})", cell_style),
                Paragraph(status_text, status_style),
                Paragraph(f"{conf:.0f}%", cell_style),
                Paragraph(score_text, cell_style),
                Paragraph(sev_lvl, cell_bold),
                Paragraph(reason, cell_style)
            ])

    defects_table = Table(table_rows, colWidths=[22, 40, 96, 48, 34, 58, 50, 192])
    defects_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(defects_table)
    story.append(Spacer(1, 14))

    # Methodology & Academic Disclaimer
    story.append(Paragraph("Engineering & Methodology Disclaimer", section_heading))
    disclaimer_text = (
        "<b>Notice on Monocular Visual Severity Estimation:</b> "
        "ROADGUARD AI computes an empirical visual severity index (0 - 100) based on detected defect type, "
        "normalized bounding box area relative to frame resolution, lane perspective proximity, and multi-frame "
        "temporal persistence. Physical road defect dimensions (such as pothole depth in millimeters or calibrated "
        "crack width in centimeters) cannot be directly determined without calibrated stereo camera geometry, "
        "LiDAR, or specialized pavement depth profiling instrumentation. Detection confidence represents model "
        "classification certainty and is maintained strictly independent of physical hazard severity. "
        "Use this report as a prioritized screening tool for road maintenance scheduling."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    return output_path
