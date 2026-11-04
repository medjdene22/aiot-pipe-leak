#!/usr/bin/env python3
"""Render 01_Rapport/report.md -> 01_Rapport/report.pdf (reportlab).

Supports a pragmatic Markdown subset: #/##/### headings, paragraphs, bullet
lists, numbered lists, GFM tables, **bold**, *italic*, `code`, > blockquotes
(used for the "Limits of the simulation" box), --- rules, and ![alt](img.png)
figures. Two-column IEEE-like layout.
"""
import os
import re
import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)
from reportlab.platypus.flowables import HRFlowable

HERE = os.path.dirname(os.path.abspath(__file__))

styles = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=16,
                            leading=21, alignment=TA_CENTER, spaceAfter=12),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13,
                         leading=17, spaceBefore=14, spaceAfter=7,
                         textColor=colors.HexColor("#0b3d6b")),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11,
                         leading=15, spaceBefore=10, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=11,
                           leading=16.5, alignment=TA_JUSTIFY, spaceAfter=10),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=11,
                             leading=16, leftIndent=14, bulletIndent=4,
                             spaceAfter=6),
    "quote": ParagraphStyle("quote", fontName="Helvetica-Oblique", fontSize=9.5,
                            leading=13.5, leftIndent=10, rightIndent=10,
                            backColor=colors.HexColor("#fff6e5"),
                            borderPadding=8, spaceAfter=8, spaceBefore=6),
    "table": ParagraphStyle("table", fontName="Helvetica", fontSize=8.5,
                            leading=11),
    "tableb": ParagraphStyle("tableb", fontName="Helvetica-Bold", fontSize=8.5,
                             leading=11),
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=9,
                          leading=12, alignment=TA_CENTER, spaceAfter=10),
    "ref": ParagraphStyle("ref", fontName="Helvetica", fontSize=9, leading=12.5,
                          spaceAfter=4),
}

MARGIN = 1.7 * cm
PAGE_W = A4[0] - 2 * MARGIN
COL_W = PAGE_W * 0.62
GAP = 0.55 * cm
MARGIN = 1.7 * cm


def inline(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', s)
    return s


def md_to_flowables(md):
    lines = md.splitlines()
    flow = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if m:
                alt, path = m.groups()
                p = os.path.join(HERE, path)
                if os.path.exists(p):
                    from PIL import Image as PILImage
                    iw, ih = PILImage.open(p).size
                    w = PAGE_W if iw / ih > 1.4 else COL_W
                    w = min(w, PAGE_W)
                    flow.append(Image(p, width=w, height=w * ih / iw))
                    flow.append(Paragraph(inline(alt), styles["cap"]))
            i += 1
        elif line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i].replace("&", "&amp;")
                            .replace("<", "&lt;").replace(">", "&gt;"))
                i += 1
            i += 1  # closing fence
            flow.append(Paragraph("<br/>".join(code),
                                  ParagraphStyle("code", fontName="Courier",
                                                 fontSize=7.2, leading=9,
                                                 backColor=colors.HexColor("#f4f4f4"),
                                                 borderPadding=4, spaceAfter=6)))
        elif line.startswith("# "):
            flow.append(Paragraph(inline(line[2:]), styles["title"]))
            i += 1
        elif line.startswith("## "):
            flow.append(Paragraph(inline(line[3:]), styles["h1"]))
            i += 1
        elif line.startswith("### "):
            flow.append(Paragraph(inline(line[4:]), styles["h2"]))
            i += 1
        elif line.startswith("> "):
            quote = []
            while i < len(lines) and lines[i].startswith("> "):
                quote.append(lines[i][2:])
                i += 1
            flow.append(Paragraph(inline(" ".join(quote)), styles["quote"]))
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                rows.append(cells)
                i += 1
            if len(rows) >= 2 and all(re.fullmatch(r":?-+:?", c) for c in rows[1]):
                rows.pop(1)
            data = [[Paragraph(inline(c), styles["tableb" if ri == 0 else "table"])
                     for c in row] for ri, row in enumerate(rows)]
            ncol = max(len(r) for r in rows)
            t = Table(data, colWidths=[(PAGE_W - 0.4 * cm) / ncol] * ncol,
                      repeatRows=1)
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe9f6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            flow.append(t)
            flow.append(Spacer(1, 6))
        elif re.match(r"^\d+\. ", line):
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                m = re.match(r"^(\d+)\. (.*)", lines[i])
                flow.append(Paragraph(inline(m.group(2)), styles["bullet"],
                                      bulletText=m.group(1) + "."))
                i += 1
        elif line.startswith("- "):
            while i < len(lines) and lines[i].startswith("- "):
                flow.append(Paragraph(inline(lines[i][2:]), styles["bullet"],
                                      bulletText="\u2022"))
                i += 1
        elif line.strip() == "---":
            flow.append(HRFlowable(width="100%", thickness=0.6,
                                   color=colors.grey, spaceAfter=6))
            i += 1
        elif line.strip() == "":
            i += 1
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(
                    ("#", "- ", "|", "> ", "!", "```")) and not re.match(r"^\d+\. ", lines[i]):
                para.append(lines[i])
                i += 1
            if para:
                flow.append(Paragraph(inline(" ".join(para)), styles["body"]))
    return flow


class TwoColumnDoc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(path, pagesize=A4,
                         leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN, bottomMargin=MARGIN)
        fw = PAGE_W
        fl = A4[1] - 2 * MARGIN
        f1 = Frame(MARGIN, MARGIN, fw, fl, id="col1")
        self.addPageTemplates([PageTemplate(id="one", frames=[f1])])


def load_results_tokens():
    """Map {{TOKENS}} in report.md to measured values from results/."""
    res = os.path.abspath(os.path.join(HERE, "..", "results"))
    tok = {}

    def j(name):
        try:
            with open(os.path.join(res, name)) as f:
                return json.load(f)
        except Exception:
            return {}

    edge = j("edge_detector_metrics.json")
    fog = j("fog_localizer_metrics.json")
    x5 = j("experiments_x5.json")
    x10 = j("experiments_x10.json")
    tok["X1_F1"] = str(edge.get("rf_edge", {}).get("f1", "NOT RUN"))
    tok["X1_ACC"] = str(edge.get("rf_edge", {}).get("accuracy", "NOT RUN"))
    tok["X1_PREC"] = str(edge.get("rf_edge", {}).get("precision", "NOT RUN"))
    tok["X1_REC"] = str(edge.get("rf_edge", {}).get("recall", "NOT RUN"))
    tok["X1_BASE_F1"] = str(edge.get("baseline_threshold", {}).get("test", {}).get("f1", "NOT RUN"))
    tok["X1_THETA"] = str(edge.get("baseline_threshold", {}).get("theta", "-"))
    tok["X1_REPLAY"] = str(edge.get("replay", {}).get("replay_accuracy", "NOT RUN"))
    tok["X1_CACC"] = str(j("consistency_check.json").get("c_model_accuracy_vs_labels", "NOT RUN"))
    tok["X1_MISMATCH"] = str(j("consistency_check.json").get("pc_vs_c_mismatches", "NOT RUN"))
    tok["X2_SRC_KB"] = str(edge.get("export", {}).get("header_kb", "NOT RUN"))
    tok["X3_PC_US"] = str(edge.get("pc_inference_us_per_window", "NOT RUN"))
    sweep = fog.get("sensor_sweep", {})
    a = sweep.get("all", {})
    tok["X4_BASE_ALL"] = str(a.get("baseline_cosine", {}).get("top1", "NOT RUN"))
    tok["X4_RF_ALL"] = str(a.get("rf", {}).get("top1", "NOT RUN"))
    tok["X4_RF3_ALL"] = str(a.get("rf", {}).get("top3", "NOT RUN"))
    tok["X4_MLP_ALL"] = str(a.get("mlp", {}).get("top1", "NOT RUN"))
    tok["X4_MLP3_ALL"] = str(a.get("mlp", {}).get("top3", "NOT RUN"))
    tok["X4_ATD_RF"] = str(a.get("rf", {}).get("atd", "NOT RUN"))
    tok["X4_ATD_MLP"] = str(a.get("mlp", {}).get("atd", "NOT RUN"))
    tok["X4_BASE_K2"] = str(sweep.get("2", {}).get("baseline_cosine", {}).get("top1", "NOT RUN"))
    tok["X4_MLP_K2"] = str(sweep.get("2", {}).get("mlp", {}).get("top1", "NOT RUN"))
    tok["X4_BASE_K5"] = str(sweep.get("5", {}).get("baseline_cosine", {}).get("top1", "NOT RUN"))
    tok["X4_MLP_K5"] = str(sweep.get("5", {}).get("mlp", {}).get("top1", "NOT RUN"))
    abl = fog.get("leak_size_ablation", {})
    tok["X4_SMALL"] = str(abl.get("small_1_3.3pct", {}).get("top1", "NOT RUN"))
    tok["X4_LARGE"] = str(abl.get("large_6.7_10pct", {}).get("top1", "NOT RUN"))
    sm = x5.get("summary_ms", {})
    comp = x5.get("components_ms", {})
    for scen, tag in (("S1 all-Cloud", "S1"), ("S2 Fog", "S2"),
                      ("S3 Edge-triggered-Fog", "S3"), ("S4 dynamic hybrid", "S4")):
        d = sm.get(scen, {})
        tok[f"X5_{tag}_MEAN"] = str(d.get("mean", "NOT RUN"))
        tok[f"X5_{tag}_STD"] = str(d.get("std", "NOT RUN"))
    tok["X5_MQTT"] = str(comp.get("mqtt_oneway_mean", "NOT RUN"))
    tok["X5_REST"] = str(comp.get("rest_rtt_mean", "NOT RUN"))
    tok["X5_FOGC"] = str(comp.get("fog_compute_mean", "NOT RUN"))
    tok["X5_CLOUDC"] = str(comp.get("cloud_compute_mean", "NOT RUN"))
    tok["X10_PSI"] = str(x10.get("psi_drifted", {}).get("drift_score", "NOT RUN"))
    tok["X10_OLD"] = str(x10.get("fog_top1_old_model_on_drift", "NOT RUN"))
    tok["X10_NEW"] = str(x10.get("fog_top1_retrained_on_drift", "NOT RUN"))

    def csv_tok(name, key_col, key_val, val_col, token):
        try:
            df = pd.read_csv(os.path.join(res, name))
            v = df[df[key_col] == key_val][val_col].iloc[0]
            tok[token] = str(v)
        except Exception:
            tok[token] = "NOT RUN"

    for scen, tag in (("S1 all-Cloud", "S1"), ("S2 Fog", "S2"),
                      ("S3 Edge-triggered-Fog", "S3"), ("S4 dynamic hybrid", "S4")):
        csv_tok("experiments_x8.csv", "scenario", scen, "mAh_per_day", f"X8_{tag}")
        csv_tok("experiments_x6.csv", "scenario", scen, "bytes_per_min", f"X6_{tag}")
    return tok


def main():
    md = open(os.path.join(HERE, "report.md")).read()
    for k, v in load_results_tokens().items():
        md = md.replace("{{" + k + "}}", v)
    import re as _re
    md = _re.sub(r"\{\{[A-Z0-9_]+\}\}", "NOT RUN", md)
    out = os.path.join(HERE, "report.pdf")
    doc = TwoColumnDoc(out)
    doc.build(md_to_flowables(md))
    print("wrote", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
