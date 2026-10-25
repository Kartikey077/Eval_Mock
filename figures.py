"""Find the diagrams / pictures that belong to each question and cut them out of the PDF page,
so they can be shown with the question (diagram in the question, or pictures used as options A-D).

What goes into the cut-out picture:
  1. normally ONLY the figure (everything else in the cut is painted white);
  2. if the figure touches the question text, the question text is included too (never the options);
  3. only if the figure touches the options, the options are included as well.
An answer / explanation / solution line is never included, and nothing below it either.

Works on pages with a real text layer (it needs text positions). Scanned pages are skipped.
Assumes the usual single-column exam layout.
"""
import base64
import io
import re
from collections import Counter

import pymupdf
from PIL import Image, ImageDraw

from quizparse import RE_LABELONLY, RE_RANGE, junk_option

RE_START = re.compile(
    r"^\s*(?:Question\s*Number\s*:\s*(\d+)|Q(?:uestion)?\s*\.?\s*(\d+)\b|(\d+)\s*[\.\):\-]\s)", re.I)
RE_LABEL = re.compile(r"^(?:\(([A-Ea-e])\)|([A-Ea-e])[\.\)])$")  # a lone "(A)", "A." or "A)"
RE_OPTLINE = re.compile(r"^\s*(?:\(\s*[A-Ea-e]\s*\)|[A-Ea-e][\.\)])(?:\s|$)")  # a line that starts an option
# "Answer: B", "Ans. C", "Correct answer - 5"   (but NOT the blank in the question: "Answer: ________")
RE_ANSLINE = re.compile(
    r"^\s*(?:correct\s+(?:answers?|options?)|answers?|ans)\s*(?:[:\-=]|\.\s)\s*(?![_\.\s]*$)\S", re.I)
RE_SOLLINE = re.compile(r"^\s*(?:explanation|solution|sol\.?|reason|hint)\s*[:\-]", re.I)

DPI = 120
MAX_FIGURES = 80
TOP_MARGIN, BOTTOM_MARGIN = 40, 35  # skip page headers/footers on continuation pages
SIDE = 30  # text this close to the figure sideways is treated as part of it (axis labels, gate names)


# ---------------------------------------------------------------- reading the page

def _norm(t):
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", t.strip().lower()))


class Pages:
    """Caches per-page word lists, text lines, rules and drawings so each page is read only once."""

    def __init__(self, doc):
        self.doc = doc
        self._w, self._d, self._l, self._h, self._m, self._furn = {}, {}, {}, {}, {}, None

    def words(self, pn):
        if pn not in self._w:
            self._w[pn] = self.doc[pn].get_text("words")
        return self._w[pn]

    def drawings(self, pn):
        if pn not in self._d:
            self._d[pn] = self.doc[pn].get_drawings()
        return self._d[pn]

    def hrules(self, pn):
        """Horizontal rules (table borders, separators): [(y, x0, x1)]."""
        if pn not in self._h:
            self._h[pn] = [((d["rect"].y0 + d["rect"].y1) / 2, d["rect"].x0, d["rect"].x1)
                           for d in self.drawings(pn) if d["rect"].height < 3 and d["rect"].width > 40]
        return self._h[pn]

    def _page_lines(self, pn):
        if pn not in self._l:
            groups = {}
            for w in self.words(pn):
                groups.setdefault((w[5], w[6]), []).append(w)
            out = []
            for ws in groups.values():
                ws.sort(key=lambda w: w[7])
                text = " ".join(w[4] for w in ws).strip()
                if text:
                    out.append({"x0": min(w[0] for w in ws), "y0": min(w[1] for w in ws),
                                "x1": max(w[2] for w in ws), "y1": max(w[3] for w in ws), "text": text,
                                "key": (ws[0][5], ws[0][6])})
            out.sort(key=lambda l: (round(l["y0"]), l["x0"]))
            self._l[pn] = out
        return self._l[pn]

    def mono(self, pn):
        """Boxes of text lines set in a monospace font (program code): in monospace text every character has
        the same width. Font names are not reliable (many PDFs only say 'CIDFont+F7')."""
        if pn not in self._m:
            boxes = []
            for b in self.doc[pn].get_text("rawdict")["blocks"]:
                if b.get("type") != 0:
                    continue
                for l in b["lines"]:
                    ws = [c["bbox"][2] - c["bbox"][0] for sp in l["spans"] for c in sp["chars"] if c["c"].strip()]
                    if len(ws) >= 5 and max(ws) - min(ws) < 0.6:
                        boxes.append(tuple(l["bbox"]))
            self._m[pn] = boxes
        return self._m[pn]

    def is_mono(self, pn, l):
        cx, cy = (l["x0"] + l["x1"]) / 2, (l["y0"] + l["y1"]) / 2
        return any(b[0] - 1 <= cx <= b[2] + 1 and b[1] - 1 <= cy <= b[3] + 1 for b in self.mono(pn))

    def _in_margin(self, pn, l):
        H = self.doc[pn].rect.height
        return l["y1"] < 0.16 * H or l["y0"] > 0.88 * H

    def furniture(self):
        """Running headers / footers: longer lines in the top or bottom margin that repeat on many pages
        ('Page 12 of 41', 'GATE 2022 Computer Science ...'). Returned as normalised texts."""
        if self._furn is None:
            cnt, n = Counter(), len(self.doc)
            for pn in range(n):
                seen = set()
                for l in self._page_lines(pn):
                    if (self._in_margin(pn, l) and len(l["text"]) >= 12
                            and not RE_START.match(l["text"]) and not RE_OPTLINE.match(l["text"])):
                        seen.add(_norm(l["text"]))
                cnt.update(seen)
            self._furn = {t for t, c in cnt.items() if c >= max(3, int(0.4 * n))}
        return self._furn

    def lines(self, pn, top, bot):
        """Text lines whose centre is between top and bot, in reading order (running headers / footers removed):
        dicts with x0, y0, x1, y1, text (words joined by single spaces)."""
        furn = self.furniture()
        return [l for l in self._page_lines(pn)
                if top <= (l["y0"] + l["y1"]) / 2 <= bot
                and not (furn and self._in_margin(pn, l) and _norm(l["text"]) in furn)]


def _cell_top(pg, pn, lab, lines):
    """Top of the question. Normally the label's own line. In question-sheet PDFs (every question in a bordered
    table) the label 'Q.8' can sit lower than the first line of its cell, so that line would be lost:
    take text to the right of the label that starts higher, up to the cell's top border."""
    y = lab["y0"]
    for m in lines:  # text to the right that starts higher but overlaps the label's row
        if m["x0"] >= lab["x1"] - 1 and m["y0"] < lab["y0"] - 0.5 and m["y1"] > lab["y0"] + 0.5:
            y = min(y, m["y0"])
    rules = [r for r in pg.hrules(pn)
             if r[1] <= lab["x0"] + 2 and r[2] >= lab["x1"] - 2 and lab["y0"] - 160 < r[0] < lab["y0"] - 2]
    if rules:
        ry = max(r[0] for r in rules)
        between = [m for m in lines if ry - 1 <= m["y0"] < lab["y0"] - 0.5]
        # lines in the label's own column (a previous label such as '(D)') mean this is not one single cell
        if between and not any(abs(m["x0"] - lab["x0"]) < 8 for m in between):
            right = [m["y0"] for m in between if m["x0"] >= lab["x1"] - 1]
            if right:
                y = min(y, min(right))
    return y


def _starts(doc, ids, skip):
    """Where does each question begin?  {position in list: (page_no, y)}.  Walks the PDF in order and only
    accepts the next expected question numbers, so numbers inside options or tables are not mistaken.
    Paper numbers can repeat (e.g. a General Aptitude part and a subject part both start at Q.1),
    so results are keyed by position, never by number."""
    pg = Pages(doc)
    found, k = {}, 0
    for pn in range(len(doc)):
        if pn in skip:
            continue
        lines = pg.lines(pn, -1, 1e6)
        for l in lines:
            t = l["text"]
            if RE_RANGE.match(t):
                continue
            m = RE_START.match(t)
            if not m:
                continue
            n = int(next(g for g in m.groups() if g))
            window = ids[k:k + 3]
            if n in window:
                found[k + window.index(n)] = (pn, _cell_top(pg, pn, l, lines))
                k += window.index(n) + 1
                if k >= len(ids):
                    return found
    return found


def question_bands(doc, qs, skip=()):
    """{position: [(page_no, top, bottom), ...]} - the page area(s) each question occupies, and the
    start positions. The area ends before the next question, and before any answer/solution line."""
    skip = set(skip)
    starts = _starts(doc, [q["id"] for q in qs], skip)
    pg = Pages(doc)
    bands = {}
    for k in range(len(qs)):
        if k not in starts:
            continue
        pn, y0 = starts[k]
        nxt = starts.get(k + 1)
        H = doc[pn].rect.height
        if nxt and nxt[0] == pn:
            raw = [(pn, y0 - 2, nxt[1] - 2)]
        elif nxt:
            raw = [(pn, y0 - 2, H - BOTTOM_MARGIN)]
            for p in range(pn + 1, min(nxt[0], pn + 2) + 1):  # at most one continuation page
                if p in skip:
                    break
                raw.append((p, TOP_MARGIN, (nxt[1] - 2) if p == nxt[0] else doc[p].rect.height - BOTTOM_MARGIN))
        else:
            raw = [(pn, y0 - 2, H - BOTTOM_MARGIN)]
        out = []
        for p, top, bot in raw:  # stop at an answer / solution line: nothing after it belongs to the question
            cut = next((l["y0"] - 1 for l in pg.lines(p, top, bot) if l["y0"] > top + 6
                        and (RE_ANSLINE.match(l["text"]) or RE_SOLLINE.match(l["text"]))), None)
            out.append((p, top, bot if cut is None else cut))
            if cut is not None:
                break
        bands[k] = out
    return starts, bands


def _repeated_images(doc):
    """Logos / watermarks that repeat on many pages are not question figures."""
    cnt = Counter()
    for page in doc:
        for im in page.get_images(full=True):
            cnt[im[0]] += 1
    return {x for x, c in cnt.items() if c >= max(3, len(doc) // 2)}


def _axis_aligned(d):
    """True when a drawing consists only of horizontal / vertical lines and rectangles (frames, borders)."""
    for it in d["items"]:
        if it[0] == "c":
            return False
        if it[0] == "l" and abs(it[1].x - it[2].x) > 0.5 and abs(it[1].y - it[2].y) > 0.5:
            return False
        if it[0] == "qu" and not it[1].is_rectangular:
            return False
    return True


def _graphics(page, drawings, ytop, ybot, marks):
    """Pictures and diagram strokes whose centre lies between ytop and ybot: [(rect, weight)].
    weight > 0 means 'looks like part of a diagram' (picture, curve, diagonal line, shaded box)."""
    W, H = page.rect.width, page.rect.height
    out = []
    for im in page.get_images(full=True):
        if im[0] in marks:
            continue
        for r in page.get_image_rects(im[0]):
            if r.width < 25 or r.height < 25 or (r.width > 0.9 * W and r.height > 0.8 * H):
                continue
            if ytop - 3 <= (r.y0 + r.y1) / 2 <= ybot + 3:
                out.append((r, 10))
    for d in drawings:
        r = d["rect"]
        cy = (r.y0 + r.y1) / 2
        if not (ytop - 3 <= cy <= ybot + 3):
            continue
        if r.width > 0.85 * W and r.height < 6:  # horizontal rules / borders
            continue
        if r.width > 0.9 * W and r.height > 0.8 * H:  # page frame
            continue
        if min(r.width, r.height) < 3 and max(r.width, r.height) > 60 and (cy < 65 or cy > H - 65):
            continue  # header / footer rule, not part of a diagram
        fill = d.get("fill")
        dark = fill is not None and all(c < 0.35 for c in fill)
        axis = _axis_aligned(d)
        if axis and fill is not None and all(c > 0.95 for c in fill):
            continue  # white box = background of a cell
        if axis and min(r.width, r.height) < 3 and max(r.width, r.height) >= 120:
            continue  # a long straight rule = table border
        if axis and dark and max(r.width, r.height) >= 150:
            continue  # a big frame drawn as a black shape = table / cell border
        w = 0
        for it in d["items"]:
            if it[0] == "c":  # curve: circles, arcs, plots
                w += 1
            elif it[0] == "l" and abs(it[1].x - it[2].x) > 1.5 and abs(it[1].y - it[2].y) > 1.5:  # diagonal line
                w += 1
        if fill is not None and tuple(fill) != (1, 1, 1) and r.width > 2 and r.height > 2:
            if not (axis and dark and (r.width <= 8 or r.height <= 8)):  # thin black bars are lines, not shapes
                w += 1
        out.append((r, w))
    return out


def _clusters(items, lines):
    """Group graphics into separate figures: pieces stay together unless real text lies between them
    (the (A)(B)(C)(D) labels of picture options do not count as text). Returns [(Rect, weight)].
    Plain lines and outlines (weight 0) only join a figure when they lie right next to it, so table borders
    far away do not stretch the figure over the question text."""
    cores = [(r, w) for r, w in items if w > 0]
    extras = [r for r, w in items if w == 0]
    out = []
    for r, w in sorted(cores, key=lambda it: it[0].y0):
        if out:
            F, tw = out[-1]
            gap_text = any(F.y1 < (l["y0"] + l["y1"]) / 2 < r.y0 and len(l["text"]) >= 20  # a real sentence, not "lim" or "1 - e"
                           and not RE_LABELONLY.match(l["text"]) for l in lines)
            if r.y0 <= F.y1 or not gap_text:
                F.include_rect(r)
                out[-1] = (F, tw + w)
                continue
        out.append((pymupdf.Rect(r), w))
    res = []
    for F, tw in out:
        if tw < 3:
            continue
        near = F + (-40, -40, 40, 40)
        for r in extras:
            if r.intersects(near) and r.width <= 1.5 * F.width + 60 and r.height <= 1.5 * F.height + 60:
                F.include_rect(r)
        res.append((F, tw))
    return res


# ---------------------------------------------------------------- cutting the figure out

def _crop(page, F, lines, top, bot, picture_labels):
    """Render the area around figure F. Returns a PNG (bytes).
    lines: the text lines in this band. Stage 1: figure only. Stage 2: + question text. Stage 3: + options."""
    W = page.rect.width
    zone = pymupdf.Rect(F.x0 - SIDE, F.y0 - 14, F.x1 + SIDE, F.y1 + 14)  # text fully inside = part of the figure
    core = pymupdf.Rect(F.x0 - 2, F.y0 - 2, F.x1 + 2, F.y1 + 2)

    def hits(l, r):
        return l["y1"] > r.y0 and l["y0"] < r.y1 and l["x1"] > r.x0 and l["x0"] < r.x1

    def inside(l):
        if l["x0"] >= zone.x0 and l["x1"] <= zone.x1 and l["y0"] >= zone.y0 and l["y1"] <= zone.y1:
            return True
        cx, cy = (l["x0"] + l["x1"]) / 2, (l["y0"] + l["y1"]) / 2  # short text centred on the figure ("lim", "1 - e")
        return len(l["text"]) < 25 and zone.x0 <= cx <= zone.x1 and zone.y0 <= cy <= zone.y1

    kind = {}  # id(line) -> 'fig' | 'opt' | 'ans' | 'stem'
    for l in lines:
        t = l["text"]
        if RE_ANSLINE.match(t) or RE_SOLLINE.match(t):
            kind[id(l)] = "ans"
        elif picture_labels and RE_LABELONLY.match(t):
            kind[id(l)] = "fig"  # (A) (B) (C) (D) labels of picture options belong to the picture
        elif RE_OPTLINE.match(t):
            kind[id(l)] = "opt"
        elif inside(l):
            kind[id(l)] = "fig"
        else:
            kind[id(l)] = "stem"

    opt_hit = any(kind[id(l)] == "opt" and hits(l, core) for l in lines)
    stem_hit = any(kind[id(l)] == "stem" and hits(l, core) for l in lines)
    stage = 3 if opt_hit else 2 if stem_hit else 1

    padt, padb = (22, 28) if picture_labels else (6, 6)
    stem_lines = [l for l in lines if kind[id(l)] == "stem"]
    opt_lines = [l for l in lines if kind[id(l)] == "opt"]
    y0, y1 = F.y0 - padt, F.y1 + padb
    x0, x1 = max(20, F.x0 - SIDE), min(W - 20, F.x1 + SIDE)
    if stage >= 2:  # include the question text above the figure
        y0 = min(y0, min([l["y0"] for l in stem_lines] or [y0]) - 2)
        x0, x1 = 20, W - 20
    if stage == 3:  # include the options
        y1 = max(y1, max([l["y1"] for l in opt_lines] or [y1]) + 3)
    figl = [l for l in lines if kind[id(l)] == "fig" and l["y1"] > y0 - 2 and l["y0"] < y1 + 2]
    if figl and stage == 1:  # text that belongs to the figure ("lim", axis labels, (A)...) must fit inside the crop
        x0 = max(5, min([x0] + [l["x0"] - 9 for l in figl]))
        x1 = min(W - 5, max([x1] + [l["x1"] + 9 for l in figl]))
    if stage == 1:  # do not run into text above / below: shrink the margin instead
        above = [l["y1"] + 3 for l in lines if kind[id(l)] != "fig" and l["y1"] <= F.y0 + 1]  # +3: glyphs overshoot the word box
        below = [l["y0"] - 3 for l in lines if kind[id(l)] != "fig" and l["y0"] >= F.y1 - 1]
        y0 = max([y0] + [a for a in above if a <= F.y0])
        y1 = min([y1] + [b for b in below if b >= F.y1])
    clip = pymupdf.Rect(x0, max(top, y0), x1, min(bot, y1))
    if clip.height < 15 or clip.width < 15:
        return None, stage

    pix = page.get_pixmap(clip=clip, dpi=DPI)
    scale = DPI / 72.0
    hide = {"ans"} | ({"opt"} if stage < 3 else set()) | ({"stem"} if stage < 2 else set())
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    draw = ImageDraw.Draw(img)
    for l in lines:  # paint over everything that must not be visible
        if kind[id(l)] in hide:
            r = pymupdf.Rect(l["x0"] - 2, l["y0"] - 3, l["x1"] + 2, l["y1"] + 3) & clip
            if not r.is_empty:
                draw.rectangle([(r.x0 - clip.x0) * scale, (r.y0 - clip.y0) * scale,
                                (r.x1 - clip.x0) * scale, (r.y1 - clip.y0) * scale], fill=(255, 255, 255))
    out = io.BytesIO()
    img.save(out, "PNG", optimize=True)
    return out.getvalue(), stage


def find_figures(doc, qs, skip=(), starts_bands=None):
    """Returns (images, picture_opts).
    images:       {position: [data-URI png, ...]}  the diagram(s) belonging to the question at that position
                  (one picture per separate figure, top to bottom)
    picture_opts: {position: ['A','B','C','D']}    questions whose options are pictures (labels but no text)
    Position = index in qs (0-based)."""
    starts, bands = starts_bands or question_bands(doc, qs, skip)
    pg = Pages(doc)
    marks = _repeated_images(doc)
    images, picture_opts = {}, {}
    for k, q in enumerate(qs):
        if k not in bands or len(images) >= MAX_FIGURES:
            continue
        real = {L for L, t in q["options"].items() if not junk_option(t)}  # options that really are text
        crops, labels = [], set()
        for p, top, bot in bands[k]:
            page = doc[p]
            lines = pg.lines(p, top, bot)
            for F, _weight in _clusters(_graphics(page, pg.drawings(p), top, bot, marks), lines):
                band_labels = set()
                for w in pg.words(p):
                    if F.y0 - 24 <= (w[1] + w[3]) / 2 <= F.y1 + 30 and top <= (w[1] + w[3]) / 2 <= bot:
                        m = RE_LABEL.match(w[4])
                        if m:
                            band_labels.add((m.group(1) or m.group(2)).upper())
                labels |= band_labels
                picture_labels = bool(band_labels - real)  # options are pictures: keep their (A)(B)(C)(D) labels
                png, _stage = _crop(page, F, lines, top, bot, picture_labels)
                if png:
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