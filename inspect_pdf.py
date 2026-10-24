"""Diagnose a PDF:  python inspect_pdf.py yourfile.pdf
Shows what the parser understood, so you can see why the AI step was (or was not) needed."""
import sys

import pymupdf

from figures import find_figures, question_bands
from layout import find_layout
from quizparse import best_text_parse, complete, quality_ok

doc = pymupdf.open(sys.argv[1])
pages = [p.get_text("text") for p in doc]
text, qs, info = best_text_parse(doc, pages)
print(f"Pages: {len(pages)}")
print("Pages with almost no text (would need OCR):", [i + 1 for i, t in enumerate(pages) if len(t.strip()) < 30] or "none")
print("\n--- first 30 lines the parser used ---")
for l in [l for l in text.splitlines() if l.strip()][:30]:
    print(repr(l))
bad = [q["id"] for q in qs if not complete(q)]
print(f"\n--- parsed {len(qs)} questions; without answer/options: {len(bad)} ---")
print("Types:", {t: sum(1 for q in qs if q["type"] == t) for t in ("MCQ", "MSQ", "NAT")})
print("Answer key:", info)
for q in qs[:5]:
    print(q["id"], q["type"], "| options:", len(q["options"]), "| answer:", q["answer"], "|", q["question"][:50])
sb = question_bands(doc, qs)
imgs, pic = find_figures(doc, qs, starts_bands=sb)
lay = find_layout(doc, qs, starts_bands=sb, images=imgs)
print("\nQuestions with a diagram/picture (screen numbers):", [i + 1 for i in sorted(imgs)] or "none",
      "| picture options:", [i + 1 for i in sorted(pic)] or "none")
print("Questions with a table / code / list layout (screen numbers):", [i + 1 for i in sorted(lay)] or "none")
print("\nQuality check passes (AI NOT needed):", quality_ok(qs))