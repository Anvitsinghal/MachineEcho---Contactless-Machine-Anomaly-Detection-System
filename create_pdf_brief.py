"""MachineEcho -- PDF Brief Description Generator.

Generates MachineEcho_Brief_Description.pdf based on docs/brief_description.md.

Usage:
    python create_pdf_brief.py
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle

def create_pdf():
    out_dir = Path(__file__).resolve().parent / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "MachineEcho_Brief_Description.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#6D28D9'),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4B5563'),
        spaceAfter=12
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1F2937'),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#374151'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        spaceAfter=4
    )

    story = []

    # Title
    story.append(Paragraph("MachineEcho", title_style))
    story.append(Paragraph("Contactless Machine Anomaly Detection on Snapdragon PCs", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#8B5CF6'), spaceAfter=12))

    # Problem Statement
    story.append(Paragraph("1. Problem Statement", h2_style))
    story.append(Paragraph(
        "Everyday machines — ceiling fans, water pumps, washing machines, compressors, and small motors — "
        "often develop subtle abnormal acoustic and vibrational patterns long before catastrophic failure. "
        "Detecting these early usually requires expensive specialized hardware sensors, dedicated technicians, or periodic teardown inspections. "
        "Consequently, predictive maintenance remains largely inaccessible for resource-constrained households, small workshops, and local businesses.",
        body_style
    ))

    # Proposed Solution
    story.append(Paragraph("2. Proposed Solution", h2_style))
    story.append(Paragraph(
        "<b>MachineEcho</b> turns an ordinary Snapdragon-powered HP laptop into a contactless machine-health monitoring device "
        "using its built-in microphone and on-device AI. The system operates in three simple phases:",
        body_style
    ))
    story.append(Paragraph("• <b>Learn Baseline:</b> Records 30–60 seconds of a machine's normal acoustic signature to train a custom baseline.", bullet_style))
    story.append(Paragraph("• <b>Real-Time Monitoring:</b> Continuously evaluates live audio windows (0.96s) accelerated via Snapdragon Hexagon NPU.", bullet_style))
    story.append(Paragraph("• <b>Interpretable Warning:</b> Outputs real-time status: <i>Normal</i>, <i>Anomaly Detected</i>, or <i>High-Risk Anomaly</i>.", bullet_style))

    # Technical Architecture
    story.append(Paragraph("3. Technical Architecture & Innovation", h2_style))
    story.append(Paragraph(
        "• <b>Audio Pipeline:</b> Built-in Mic (16 kHz mono) → Log-Mel Spectrogram (96×64 patch).<br/>"
        "• <b>Feature Extractor:</b> YAMNet ONNX model (Qualcomm AI Hub) producing 1024-dim embeddings.<br/>"
        "• <b>Anomaly Model:</b> One-Class Isolation Forest trained exclusively on normal operation data.<br/>"
        "• <b>Calibrated Scoring:</b> Z-score normalization converts raw scores to intuitive [0, 1] anomaly metric.<br/>"
        "• <b>Zero Cold-Start:</b> Machine-agnostic architecture capable of inspecting novel appliances without retrain datasets.",
        body_style
    ))

    # Why Snapdragon
    story.append(Paragraph("4. Why Snapdragon?", h2_style))
    story.append(Paragraph(
        "• <b>Dedicated NPU Execution:</b> Uses Qualcomm's Hexagon NPU for sub-millisecond inference, preventing CPU thermal throttling.<br/>"
        "• <b>100% On-Device & Private:</b> All acoustic analysis remains local on the PC. Zero audio streaming to cloud servers.<br/>"
        "• <b>Qualcomm AI Hub Integration:</b> Pre-optimized YAMNet model profiles target Snapdragon X Elite / X Plus architecture.<br/>"
        "• <b>Low Power Consumption:</b> Enables hours of continuous contactless monitoring on battery power.",
        body_style
    ))

    # Key Results
    story.append(Paragraph("5. Empirical Verification Results", h2_style))
    
    table_data = [
        ["Condition", "Mean Anomaly Score", "Status", "Detection Recall"],
        ["Normal Operation", "0.1367", "NORMAL [OK]", "Baseline Match"],
        ["Abnormal Operation", "0.6195", "ANOMALY DETECTED [!!]", "28 / 30 Windows (93.3%)"]
    ]
    t = Table(table_data, colWidths=[130, 120, 140, 140])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F3E8FF')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#6D28D9')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#DDD6FE')),
        ('ALIGN', (1,0), (1,-1), 'CENTER'),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t)

    # Footer note
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E5E7EB'), spaceBefore=10, spaceAfter=8))
    story.append(Paragraph("MachineEcho -- Contactless Machine Anomaly Detection on Snapdragon PCs | Author: Anvit Singhal", ParagraphStyle('Footer', parent=body_style, fontSize=8, textColor=colors.HexColor('#9CA3AF'), alignment=1)))

    doc.build(story)
    print(f"Successfully generated PDF brief -> {pdf_path}")

if __name__ == "__main__":
    create_pdf()
