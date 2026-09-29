"""MachineEcho -- PowerPoint Pitch Deck Generator.

Generates MachineEcho_Pitch_Deck.pptx based on docs/pitch_deck.md.

Usage:
    python create_presentation.py
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Colors
    DARK_BG = RGBColor(15, 23, 42)       # Slate 900
    CARD_BG = RGBColor(30, 41, 59)       # Slate 800
    TEXT_LIGHT = RGBColor(241, 245, 249) # Slate 100
    TEXT_MUTED = RGBColor(148, 163, 184)# Slate 400
    ACCENT_PURPLE = RGBColor(124, 58, 237) # Vibrant Purple
    ACCENT_GREEN = RGBColor(16, 185, 129)  # Emerald Normal
    ACCENT_ORANGE = RGBColor(245, 158, 11) # Amber Warning

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = DARK_BG
        bg.line.fill.background()
        return bg

    # -------------------------------------------------------------
    # SLIDE 1: Title & Problem
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Title box
    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.333), Inches(1.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "MachineEcho"
    p1.font.size = Pt(44)
    p1.font.bold = True
    p1.font.color.rgb = ACCENT_PURPLE

    p1_sub = tf1.add_paragraph()
    p1_sub.text = "Contactless Machine Anomaly Detection on Snapdragon PCs"
    p1_sub.font.size = Pt(22)
    p1_sub.font.color.rgb = TEXT_MUTED
    p1_sub.space_before = Pt(10)

    # Card 1: Problem
    card1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(2.6), Inches(11.333), Inches(4.2))
    card1.fill.solid()
    card1.fill.fore_color.rgb = CARD_BG
    card1.line.color.rgb = CARD_BG
    tf_card1 = card1.text_frame
    tf_card1.word_wrap = True
    
    p_c1_title = tf_card1.paragraphs[0]
    p_c1_title.text = "Problem Statement: Everyday Machines Fail Without Warning"
    p_c1_title.font.size = Pt(24)
    p_c1_title.font.bold = True
    p_c1_title.font.color.rgb = TEXT_LIGHT

    bullets1 = [
        "Ceiling fans, water pumps, washing machines, compressors & small motors develop acoustic/vibrational anomalies before failure.",
        "Detecting early anomalies usually requires specialized hardware sensors, technicians, or periodic teardowns.",
        "Resource-constrained households and small workshops lack affordable predictive maintenance solutions.",
        "Core Goal: Transform an ordinary Snapdragon-powered HP laptop mic into a contactless machine-health monitoring device."
    ]

    for b in bullets1:
        pb = tf_card1.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(18)
        pb.font.color.rgb = TEXT_MUTED
        pb.space_before = Pt(12)

    # -------------------------------------------------------------
    # SLIDE 2: Solution Overview
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)

    tb2 = s2.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.333), Inches(1.2))
    tf2 = tb2.text_frame
    p2 = tf2.paragraphs[0]
    p2.text = "Solution Overview: 3-Step Contactless AI"
    p2.font.size = Pt(36)
    p2.font.bold = True
    p2.font.color.rgb = ACCENT_PURPLE

    steps = [
        ("1. LEARN", "Record 30-60s of normal machine operation. System extracts YAMNet embeddings and learns baseline signature.", ACCENT_PURPLE),
        ("2. MONITOR", "Continuously listen to the machine via laptop mic. Process 0.96s audio windows in real-time on Hexagon NPU.", ACCENT_GREEN),
        ("3. ALERT", "Detect acoustic deviations instantly. Display Normal (0.14) or Anomaly Detected (0.62) with zero cloud dependency.", ACCENT_ORANGE)
    ]

    for i, (title, desc, color) in enumerate(steps):
        x = Inches(1.0 + i * 3.9)
        c = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(2.3), Inches(3.6), Inches(4.5))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = color
        
        tfc = c.text_frame
        tfc.word_wrap = True
        
        pt = tfc.paragraphs[0]
        pt.text = title
        pt.font.size = Pt(22)
        pt.font.bold = True
        pt.font.color.rgb = color

        pd = tfc.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(16)
        pd.font.color.rgb = TEXT_MUTED
        pd.space_before = Pt(16)

    # -------------------------------------------------------------
    # SLIDE 3: Architecture & Technical Innovation
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)

    tb3 = s3.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.333), Inches(1.2))
    tf3 = tb3.text_frame
    p3 = tf3.paragraphs[0]
    p3.text = "Architecture & Technical Pipeline"
    p3.font.size = Pt(36)
    p3.font.bold = True
    p3.font.color.rgb = ACCENT_PURPLE

    card3 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(2.2), Inches(11.333), Inches(4.6))
    card3.fill.solid()
    card3.fill.fore_color.rgb = CARD_BG
    card3.line.color.rgb = CARD_BG
    tfc3 = card3.text_frame
    tfc3.word_wrap = True

    p3_head = tfc3.paragraphs[0]
    p3_head.text = "Microphone Stream -> Mel Spectrogram -> YAMNet ONNX (NPU) -> Isolation Forest -> Dashboard"
    p3_head.font.size = Pt(20)
    p3_head.font.bold = True
    p3_head.font.color.rgb = ACCENT_GREEN

    points3 = [
        "Feature Extractor: YAMNet (Qualcomm AI Hub) generates 1024-dim embeddings from log-mel spectrogram patches (96x64).",
        "Anomaly Detection: Scikit-Learn Isolation Forest trained exclusively on normal operation data (one-class learning).",
        "Calibrated Scoring: Z-score normalization maps decision scores to intuitive [0, 1] anomaly scores.",
        "Zero Cold-Start: Works out-of-the-box on novel machines without prior training datasets for every appliance type."
    ]
    for pt in points3:
        p = tfc3.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(17)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(12)

    # -------------------------------------------------------------
    # SLIDE 4: Why Snapdragon Advantage
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)

    tb4 = s4.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.333), Inches(1.2))
    tf4 = tb4.text_frame
    p4 = tf4.paragraphs[0]
    p4.text = "Why Snapdragon? Fundamental NPU Acceleration"
    p4.font.size = Pt(36)
    p4.font.bold = True
    p4.font.color.rgb = ACCENT_PURPLE

    s4_items = [
        ("Dedicated Hexagon NPU", "Runs continuous real-time audio inference (~0.78-2.66 ms) without consuming CPU/GPU budget.", ACCENT_PURPLE),
        ("Qualcomm AI Hub Support", "YAMNet is natively supported & optimized for Snapdragon X Elite / X Plus architecture.", ACCENT_GREEN),
        ("Low Power & Privacy", "All audio processing remains 100% on-device. No audio streamed to cloud, low battery drain.", ACCENT_ORANGE)
    ]

    for i, (title, desc, color) in enumerate(s4_items):
        y = Inches(2.2 + i * 1.6)
        c = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), y, Inches(11.333), Inches(1.4))
        c.fill.solid()
        c.fill.fore_color.rgb = CARD_BG
        c.line.color.rgb = color

        tfc = c.text_frame
        tfc.word_wrap = True
        pt = tfc.paragraphs[0]
        pt.text = title
        pt.font.size = Pt(20)
        pt.font.bold = True
        pt.font.color.rgb = color

        pd = tfc.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(16)
        pd.font.color.rgb = TEXT_MUTED
        pd.space_before = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 5: Live Benchmark & Verification Results
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)

    tb5 = s5.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.333), Inches(1.2))
    tf5 = tb5.text_frame
    p5 = tf5.paragraphs[0]
    p5.text = "Live Results & Empirical Validation"
    p5.font.size = Pt(36)
    p5.font.bold = True
    p5.font.color.rgb = ACCENT_PURPLE

    card5 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(2.2), Inches(11.333), Inches(4.6))
    card5.fill.solid()
    card5.fill.fore_color.rgb = CARD_BG
    card5.line.color.rgb = CARD_BG
    tfc5 = card5.text_frame
    tfc5.word_wrap = True

    r_items = [
        "Normal Fan Audio: Mean Anomaly Score = 0.1367 | Status: NORMAL [OK]",
        "Abnormal Fan Audio: Mean Anomaly Score = 0.6195 | Status: ANOMALY DETECTED [!!]",
        "Detection Accuracy: 28 out of 30 abnormal windows correctly flagged (93.3% recall).",
        "Inference Speed: Sub-millisecond latency per audio window on-device.",
        "Interactive Dashboard: Real-time gauge, rolling anomaly score chart & audio file analysis mode."
    ]

    p5_head = tfc5.paragraphs[0]
    p5_head.text = "Measured Performance on Synthetic & Recorded Datasets"
    p5_head.font.size = Pt(22)
    p5_head.font.bold = True
    p5_head.font.color.rgb = ACCENT_GREEN

    for item in r_items:
        p = tfc5.add_paragraph()
        p.text = "✔ " + item
        p.font.size = Pt(18)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 6: Roadmap & Submission Package
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)

    tb6 = s6.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.333), Inches(1.2))
    tf6 = tb6.text_frame
    p6 = tf6.paragraphs[0]
    p6.text = "Roadmap & Final Deliverables"
    p6.font.size = Pt(36)
    p6.font.bold = True
    p6.font.color.rgb = ACCENT_PURPLE

    card6 = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(2.2), Inches(11.333), Inches(4.6))
    card6.fill.solid()
    card6.fill.fore_color.rgb = CARD_BG
    card6.line.color.rgb = CARD_BG
    tfc6 = card6.text_frame
    tfc6.word_wrap = True

    roadmap = [
        "Phase 1 (Current MVP): Real-time acoustic anomaly detection with YAMNet & Isolation Forest.",
        "Phase 2 (Next): Multimodal camera-assisted visual inspection (detecting physical oscillations/defects).",
        "Phase 3: Multi-machine simultaneous acoustic monitoring for small workshops & home appliances.",
        "Deliverables Ready: Working Streamlit App, Modular Codebase, Benchmark Suite, Pitch Deck & Brief."
    ]

    p6_head = tfc6.paragraphs[0]
    p6_head.text = "Project Status: Production-Ready MVP & Documented Submission Package"
    p6_head.font.size = Pt(22)
    p6_head.font.bold = True
    p6_head.font.color.rgb = ACCENT_ORANGE

    for rm in roadmap:
        p = tfc6.add_paragraph()
        p.text = "➔ " + rm
        p.font.size = Pt(18)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(12)

    # Save presentation
    out_dir = Path(__file__).resolve().parent / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "MachineEcho_Pitch_Deck.pptx"
    prs.save(str(out_path))
    print(f"Successfully generated PowerPoint presentation -> {out_path}")

if __name__ == "__main__":
    create_deck()
