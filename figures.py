"""Find the diagrams / pictures that belong to each question and cut them out of the PDF page,
so they can be shown with the question (diagram in the question, or pictures used as options A-D).

Works on pages with a real text layer (it needs text positions). Scanned pages are skipped.
Assumes the usual single-column exam layout.
"""
import base64
import re
from collections import Counter

import pymupdf

from quizparse import RE_RANGE, junk_option

RE_START = re.compile(
    r"^\s*(?:Question\s*Number\s*:\s*(\d+)|Q(?:uestion)?\s*\.?\s*(\d+)\b|(\d+)\s*[\.\):\-]\s)", re.I)
RE_LABEL = re.compile(r"^(?:\(([A-Ea-e])\)|([A-Ea-e])[\.\)])$")  # a lone "(A)", "A." or "A)"

DPI = 120
MAX_FIGURES = 80
TOP_MARGIN, BOTTOM_MARGIN = 40, 35  # skip page headers/footers on continuation pages


def _lines(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            t = "".join(s["text"] for s in l["spans"]).strip()
            if t:
                out.append((l["bbox"][1], l["bbox"][0], t))
    out.sort()
    return out


def _starts(doc, ids, skip):
    """Where does each question begin?  {position in list: (page_no, y)}.  Walks the PDF in order and only
    accepts the next expected question numbers, so numbers inside options or tables are not mistaken.
    Paper numbers can repeat (e.g. a General Aptitude part and a subject part both start at Q.1),
    so results are keyed by position, never by number."""
    found, k = {}, 0
    for pn in range(len(doc)):
        if pn in skip:
            continue
        for y, _x, t in _lines(doc[pn]):
            if RE_RANGE.match(t):
                continue
            m = RE_START.match(t)
            if not m:
                continue
            n = int(next(g for g in m.groups() if g))
            window = ids[k:k + 3]
            if n in window:
                found[k + window.index(n)] = (pn, y)
                k += window.index(n) + 1
                if k >= len(ids):
                    return found
    return found


def _repeated_images(doc):
    """Logos / watermarks that repeat on many pages are not question figures."""
    cnt = Counter()
    for page in doc:
        for im in page.get_images(full=True):
            cnt[im[0]] += 1
    return {x for x, c in cnt.items() if c >= max(3, len(doc) // 2)}


def _graphics(page, drawings, ytop, ybot, marks):
    """Pictures and diagram strokes whose centre lies between ytop and ybot. Returns (rects, score)."""
    W, H = page.rect.width, page.rect.height
    rects, score = [], 0
    for im in page.get_images(full=True):
        if im[0] in marks:
            continue
        for r in page.get_image_rects(im[0]):
            if r.width < 25 or r.height < 25 or (r.width > 0.9 * W and r.height > 0.8 * H):
                continue
            if ytop - 3 <= (r.y0 + r.y1) / 2 <= ybot + 3:
                rects.append(r)
                score += 10
    for d in drawings:
        r = d["rect"]
        if not (ytop - 3 <= (r.y0 + r.y1) / 2 <= ybot + 3):
            continue
        if r.width > 0.85 * W and r.height < 6:  # horizontal rules / borders
            continue
        if r.width > 0.9 * W and r.height > 0.8 * H:  # page frame
            continue
        cy = (r.y0 + r.y1) / 2
        if min(r.width, r.height) < 3 and max(r.width, r.height) > 60 and (cy < 65 or cy > H - 65):
            continue  # header / footer rule, not part of a diagram
        for it in d["items"]:
            if it[0] == "c":  # curve: circles, arcs, plots
                score += 1
            elif it[0] == "l" and abs(it[1].x - it[2].x) > 1.5 and abs(it[1].y - it[2].y) > 1.5:  # diagonal line
                score += 1
        fill = d.get("fill")
        if fill is not None and tuple(fill) != (1, 1, 1) and r.width > 2 and r.height > 2:
            score += 1
        rects.append(r)
    return rects, score


def find_figures(doc, qs, skip=()):
    """Returns (images, picture_opts).
    images:       {position: [data-URI png, ...]}  the diagram(s) belonging to the question at that position
    picture_opts: {position: ['A','B','C','D']}    questions whose options are pictures (labels but no text)
    Position = index in qs (0-based)."""
    ids = [q["id"] for q in qs]
    starts = _starts(doc, ids, set(skip))
    marks = _repeated_images(doc)
    drawings = {}
    images, picture_opts = {}, {}
    for k, q in enumerate(qs):
        if k not in starts or len(images) >= MAX_FIGURES:
            continue
        pn, y0 = starts[k]
        nxt = starts.get(k + 1)
        H = doc[pn].rect.height
        # the bands (page, top, bottom) this question occupies
        if nxt and nxt[0] == pn:
            bands = [(pn, y0 - 2, nxt[1] - 2)]
        elif nxt:
            bands = [(pn, y0 - 2, H - BOTTOM_MARGIN)]
            for p in range(pn + 1, min(nxt[0], pn + 2) + 1):  # at most one continuation page
                if p in skip:
                    break
                bands.append((p, TOP_MARGIN, (nxt[1] - 2) if p == nxt[0] else doc[p].rect.height - BOTTOM_MARGIN))
        else:
            bands = [(pn, y0 - 2, H - BOTTOM_MARGIN)]

        real = {L for L, t in q["options"].items() if not junk_option(t)}  # options that really are text
        crops, labels = [], set()
        for p, top, bot in bands:
            page = doc[p]
            if p not in drawings:
                drawings[p] = page.get_drawings()
            rects, score = _graphics(page, drawings[p], top, bot, marks)
            if not rects or score < 3:
                continue
            u = pymupdf.Rect(rects[0])
            for r in rects[1:]:
                u.include_rect(r)
            W = page.rect.width
            band_labels = set()
            for w in page.get_text("words", clip=pymupdf.Rect(0, top, W, bot)):
                m = RE_LABEL.match(w[4])
                if m:
                    band_labels.add((m.group(1) or m.group(2)).upper())
            labels |= band_labels
            # Picture options: keep room for the (A)(B)(C)(D) labels. Otherwise crop tight, so the
            # text of the options just below the diagram is not cut in half.
            pad_top, pad_bot = (22, 28) if band_labels - real else (6, 6)
            clip = pymupdf.Rect(20, max(top, u.y0 - pad_top), W - 20, min(bot, u.y1 + pad_bot))
            if clip.height < 15:
                continue
            png = page.get_pixmap(clip=clip, dpi=DPI).tobytes("png")
            crops.append("data:image/png;base64," + base64.b64encode(png).decode())
        if not crops:
            continue
        images[k] = crops

        # options that are pictures: labels exist in the page text but their content is not text
        union = real | labels
        if "A" in union and (real or len(union) >= 3) and union - real:
            top_letter = max(union)
            picture_opts[k] = [chr(c) for c in range(ord("A"), ord(top_letter) + 1)]
    return images, picture_opts