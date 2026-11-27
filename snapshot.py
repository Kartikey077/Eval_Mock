"""PICTURE MODE - the fallback that does not read the question, it CUTS it out.

Each question (text + maths + diagram + options, exactly as printed) becomes ONE picture. The text layer is used
only to find where a question starts and ends, never to read its content, so no option can be lost or misread.
The student sees the picture and just picks the letter(s) / types the number.

Ladder, each step used only when the one before cannot work:
  1. Text layer has "Q.5" / "Question 5" / "5)" marks  -> cut from one mark to the next (multi-page questions are stitched)
  2. Scanned pages, or no marks found                  -> cut the page at its biggest blank gaps (no OCR needed)
  3. Nothing else                                      -> the whole page is one picture
Only a completely blank PDF produces nothing.
"""
import base64
import io
import logging
import re
import statistics
from collections import Counter

import pymupdf
from PIL import Image, ImageDraw

from figures import (BOTTOM_MARGIN, RE_ANSLINE, RE_OPTLINE, RE_SOLLINE, RE_START, RE_START_NUM, TOP_MARGIN,
                     Pages, _cell_top, _norm)
from quizparse import RE_ANS, RE_META, RE_RANGE, _split_key, parse_answer

log = logging.getLogger("mocktest")
DPI = 150          # sharp enough for maths (exponents, fractions); lower makes small symbols blurry
MAX_W = 1000       # wider pictures are scaled down
MAX_Q = 300
RE_LABEL_ALONE = re.compile(r"^\s*(?:Question|Q)\s*\.?\s*\d+\s*[\.\):]?\s*$", re.I)
RE_OPTTOK = re.compile(r"\(\s*([A-Ea-e])\s*\)|\(\s*([1-5])\s*\)")
RE_OPTHEAD = re.compile(r"^\s*([A-Ea-e])[\.\)]|^\s*\(\s*[1-5]\s*\)")


def _find_starts(doc, pg, text_pages):
    """[(page, y, number)] of every question mark, in reading order."""
    def scan(numbered):
        out, prev = [], 0
        for pn in text_pages:
            W = doc[pn].rect.width
            lines = pg.lines(pn, -1, 1e6)
            for l in lines:
                t = l["text"]
                if RE_RANGE.match(t) or l["x0"] > 0.45 * W:  # a mark sits at the left; "see Question 3" does not
                    continue
                m = RE_START.match(t)
                if m:
                    n = int(next(g for g in m.groups() if g))
                else:
                    m = RE_START_NUM.match(t) if numbered else None
                    if not m:
                        continue
                    n = int(m.group(1))
                    if not (n == prev + 1 or n == 1):
                        continue
                out.append((pn, l, lines, n))
                prev = n
        return out
    s = scan(False)
    s = s if len(s) >= 2 else scan(True)
    # question-sheet PDFs (each question in a bordered cell): a cell border sits just above every label. Only then is the
    # table-aware top used; a header rule far above the page's first question must not pull its top upwards.
    near = sum(1 for pn, l, _ls, _n in s
               if any(r[1] <= l["x0"] + 2 and r[2] >= l["x1"] - 2 and l["y0"] - 30 < r[0] < l["y0"] - 1 for r in pg.hrules(pn)))
    cell = bool(s) and near >= 0.6 * len(s)
    return [(pn, _cell_top(pg, pn, l, ls) if cell else _label_top(l, ls), n) for pn, l, ls, n in s]


def _label_top(lab, lines):
    """Top of a question = its label's line (or text to the right of the label that starts a little higher)."""
    y = lab["y0"]
    for m in lines:
        if m["x0"] >= lab["x1"] - 1 and m["y0"] < lab["y0"] - 0.5 and m["y1"] > lab["y0"] + 0.5:
            y = min(y, m["y0"])
    return y


def _limits(doc, pg, pn):
    """Usable top / bottom of a page: running headers and footers are left out."""
    H = doc[pn].rect.height
    furn = pg.furniture()
    ls = pg._page_lines(pn)
    W = doc[pn].rect.width
    rules = [r for r in pg.hrules(pn) if r[2] - r[1] > 0.6 * W]  # long thin lines = header / footer separators
    top = max([TOP_MARGIN] + [l["y1"] + 1 for l in ls if l["y1"] < 0.16 * H and _norm(l["text"]) in furn]
              + [r[0] + 1 for r in rules if r[0] < 0.16 * H])
    bot = min([H - BOTTOM_MARGIN] + [l["y0"] - 1 for l in ls if l["y0"] > 0.88 * H and _norm(l["text"]) in furn]
              + [r[0] - 1 for r in rules if r[0] > 0.85 * H])
    return top, bot


def _gfx(pg, pn):
    """Real diagram parts of a page as Rects: no watermark tiles (hundreds of tiny pictures of one size), no logos in
    the margin, no page-wide rules. Cached per page."""
    cache = pg.__dict__.setdefault("_gfx_cache", {})
    if pn in cache:
        return cache[pn]
    page = pg.doc[pn]
    W = page.rect.width
    tl, bl = _limits(pg.doc, pg, pn)
    imgs = [(x, r) for x, r in pg.images(pn) if r.width >= 12 and r.height >= 8]
    sizes = Counter((round(r.width), round(r.height)) for _x, r in imgs)
    out = [r for _x, r in imgs if sizes[(round(r.width), round(r.height))] < 8 and r.width < 0.9 * W  # repeated size = tiles
           and r.y1 > tl + 6 and r.y0 < bl - 6]  # logos in the header / footer are not figures
    for d in pg.drawings(pn):
        r = d["rect"]
        if r.width > 0.85 * W or (min(r.width, r.height) < 3 and max(r.width, r.height) > 150):
            continue  # page frames, long table / header rules
        if r.y1 <= tl + 6 or r.y0 >= bl - 6:
            continue  # header / footer decoration
        out.append(r)
    if len(cache) > 8:
        cache.clear()
    cache[pn] = out
    return out


def _attached(pg, pn, y, below):
    """Diagram parts that CROSS the line y (the cut under the question, or the top mark of the next one) belong to the
    question that owns that line: the arrowhead and labels of a figure that hangs below 'Ans. : (c)' must not be lost.
    Returns the Rects (diagram pieces + their short labels) lying on the far side of y; [] when nothing crosses."""
    rects = _gfx(pg, pn)
    seeds = [r for r in rects if r.y0 < y - 0.5 and r.y1 > y + 0.5 and r.width >= 6 and r.height >= 4]
    if not seeds:
        return []
    group, union = list(seeds), pymupdf.Rect(seeds[0])
    for r in seeds:
        union.include_rect(r)
    rest = [r for r in rects if all(r is not g for g in seeds)
            and ((r.y0 < y + 160) if below else (r.y1 > y - 160))]
    grown = True
    while grown and len(group) < 400:  # pieces touching the figure are part of it (dashes, arrow heads, axes)
        grown = False
        near = union + (-6, -6, 6, 6)
        for r in rest:
            if r.intersects(near):
                group.append(r)
                union.include_rect(r)
                rest = [x for x in rest if x is not r]
                grown = True
                break
    zone = pymupdf.Rect(union.x0 - 30, union.y0 - 14, union.x1 + 30, union.y1 + 14)
    keep = [pymupdf.Rect(r) + (-1, -1, 1, 1) for r in group]
    for l in pg.lines(pn, zone.y0, zone.y1):  # labels of the figure ("(-1,0)", "Re z"), never solution text
        t = l["text"]
        if RE_ANSLINE.match(t) or RE_SOLLINE.match(t) or RE_META.search(t):
            continue
        cx, cy = (l["x0"] + l["x1"]) / 2, (l["y0"] + l["y1"]) / 2
        inside = l["x0"] >= zone.x0 and l["x1"] <= zone.x1
        if (inside or len(t) < 25) and zone.x0 <= cx <= zone.x1 and zone.y0 <= cy <= zone.y1:
            keep.append(pymupdf.Rect(l["x0"] - 2, l["y0"] - 2, l["x1"] + 2, l["y1"] + 2))
    return [k for k in keep if (k.y1 > y + 1 if below else k.y0 < y - 1)]


def _cut_at_answer(pg, pieces):
    """Stop at an 'Answer: B' / 'Solution:' line (the student must not see it). Returns (pieces, answer).
    A piece is (page, top, bottom, keep): keep = diagram parts that cross the cut and must stay visible."""
    out, ans = [], None
    for i, (p, top, bot) in enumerate(pieces):
        cut = None
        guard = top + 6 if i == 0 else top - 1  # only the first piece starts with the question's own line
        for l in pg.lines(p, top, bot):
            if l["y0"] > guard and (RE_ANSLINE.match(l["text"]) or RE_SOLLINE.match(l["text"])):
                cut = l
                break
        if cut is not None:
            m = RE_ANS.match(cut["text"])
            ans = parse_answer(m.group(1)) if m else None
            b = cut["y0"] - 1
            out.append((p, top, b, _attached(pg, p, b, True)))
            break
        out.append((p, top, bot, []))
    return out, ans


def _option_count(pg, pieces):
    """How many options does the question show? Counts (a)(b)(c).. or (1)(2)(3).. in order from A. None = unknown."""
    found = set()
    for p, top, bot, _keep in pieces:
        for l in pg.lines(p, top, bot):
            t = l["text"]
            if RE_OPTLINE.match(t) or RE_OPTHEAD.match(t):
                for m in RE_OPTTOK.finditer(t):
                    found.add(m.group(1).upper() if m.group(1) else chr(64 + int(m.group(2))))
                m = RE_OPTHEAD.match(t)
                if m and m.group(1):
                    found.add(m.group(1).upper())
    n = 0
    while chr(65 + n) in found:
        n += 1
    return n if n >= 2 else None


def _start_spec(doc, pg, starts, i, scanned):
    pn, y0, num = starts[i]
    nxt = starts[i + 1] if i + 1 < len(starts) else None
    if nxt and nxt[0] == pn:
        raw = [(pn, y0 - 2, nxt[1] - 2)]
    else:
        raw = [(pn, y0 - 2, _limits(doc, pg, pn)[1])]
        if nxt:  # question runs on to the next page(s) (at most 3): stitched into one picture
            for p in range(pn + 1, min(nxt[0], pn + 3) + 1):
                if p in scanned:
                    break
                t, b = _limits(doc, pg, p)
                raw.append((p, t, nxt[1] - 2 if p == nxt[0] else b))
                if p == nxt[0]:
                    break
    pieces, ans = _cut_at_answer(pg, raw)
    p0, t0, b0, k0 = pieces[0]  # a figure that starts above the question label (icon next to "Q.3") belongs to it
    pieces[0] = (p0, t0, b0, k0 + _attached(pg, p0, t0, False))
    return {"p": pn, "y": y0, "num": num, "pieces": pieces, "ans": ans, "k": _option_count(pg, pieces)}


def _gap_bands(page):
    """No question marks: cut the page at its biggest blank horizontal gaps. Needs no text and no OCR."""
    H = page.rect.height
    pix = page.get_pixmap(dpi=72, colorspace=pymupdf.csGRAY, alpha=False)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    pix = None
    prof = img.resize((1, img.height), Image.BOX).tobytes()  # mean brightness of every pixel row
    sy = H / img.height
    n = len(prof)
    ink = [(v < 250) and 0.05 * n < i < 0.95 * n for i, v in enumerate(prof)]  # header / footer margins ignored
    rows = [i for i, f in enumerate(ink) if f]
    if not rows:
        return []
    runs, i = [], rows[0]
    while i <= rows[-1]:
        if not ink[i]:
            j = i
            while not ink[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    th = max(9, 2 * statistics.median([b - a for a, b in runs])) if runs else 9
    cuts = [r for r in runs if r[1] - r[0] >= th]
    edges = [rows[0]] + [x for a, b in cuts for x in (a, b)] + [rows[-1] + 1]
    out = []
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a >= 12:
            out.append(((a - 3) * sy, (b + 3) * sy))
    return out


def _has_content(pg, p, top, bot):
    """Text or a diagram inside the band? (an empty strip of margin must not become a picture)"""
    if bot - top < 3:
        return False
    if pg.lines(p, top, bot):
        return True
    return any(r.y1 > top and r.y0 < bot for r in _gfx(pg, p))


def _picture(doc, pg, pieces):
    """The pieces of one question -> one PNG data-URI (None when there is nothing but white)."""
    parts = []
    scale = DPI / 72.0
    for p, top, bot, keep in pieces:
        page = doc[p]
        W = page.rect.width
        ext_top = min([top] + [k.y0 for k in keep])
        ext_bot = max([bot] + [k.y1 for k in keep])
        clip = pymupdf.Rect(0, ext_top, W, ext_bot) & page.rect
        if clip.height < 6 or (not keep and not _has_content(pg, p, top, bot)):
            continue
        pix = page.get_pixmap(clip=clip, dpi=DPI, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pix = None

        def px(r):
            return ((r.x0 - clip.x0) * scale, (r.y0 - clip.y0) * scale, (r.x1 - clip.x0) * scale, (r.y1 - clip.y0) * scale)

        if keep:  # outside [top, bot] only the diagram parts that cross the line stay; the rest (other text) is wiped
            orig = img.copy()
            d = ImageDraw.Draw(img)
            if ext_top < top:
                d.rectangle(px(pymupdf.Rect(0, ext_top, W, top)), fill=(255, 255, 255))
            if ext_bot > bot:
                d.rectangle(px(pymupdf.Rect(0, bot, W, ext_bot)), fill=(255, 255, 255))
            for k in keep:
                k = k & clip
                if not k.is_empty:
                    box = tuple(int(round(v)) for v in px(k))
                    img.paste(orig.crop(box), box[:2])
            orig.close()
        rects = [(l["x0"] - 2, l["y0"] - 3, l["x1"] + 2, l["y1"] + 3) for l in pg.lines(p, top - 1, bot + 1)
                 if RE_META.search(l["text"]) or RE_LABEL_ALONE.match(l["text"])
                 or RE_ANSLINE.match(l["text"]) or RE_SOLLINE.match(l["text"])]

        # neighbour questions: wipe only the sliver of their letters (descenders / tops) that reaches into this cut,
        # never more, so this question's own first / last line stays intact
        def mine(l):  # a label that belongs to a figure crossing the cut is not a neighbour's text
            lr = pymupdf.Rect(l["x0"], l["y0"], l["x1"], l["y1"])
            return any(lr.intersects(k) for k in keep)
        rects += [(l["x0"] - 2, l["y0"] - 3, l["x1"] + 2, min(l["y1"] + 3, top + 1.9))
                  for l in pg.lines(p, top - 20, top - 0.01) if not mine(l)]
        rects += [(l["x0"] - 2, max(l["y0"] - 3, bot - 1.2), l["x1"] + 2, l["y1"] + 3)
                  for l in pg.lines(p, bot + 0.01, bot + 20) if not mine(l)]
        if rects:
            d = ImageDraw.Draw(img)
            for x0, y0, x1, y1 in rects:
                d.rectangle(px(pymupdf.Rect(x0, y0, x1, y1)), fill=(255, 255, 255))
        parts.append(img)
    if not parts:
        return None
    if len(parts) == 1:
        img = parts[0]
    else:  # stitch pieces from several pages
        img = Image.new("RGB", (max(i.width for i in parts), sum(i.height for i in parts)), (255, 255, 255))
        y = 0
        for i in parts:
            img.paste(i, (0, y))
            y += i.height
            i.close()
    bb = img.convert("L").point(lambda v: 255 if v < 200 else 0).getbbox()  # trim the white around the content
    if not bb:
        img.close()
        return None
    pad = 8
    img = img.crop((max(0, bb[0] - pad), max(0, bb[1] - pad), min(img.width, bb[2] + pad), min(img.height, bb[3] + pad)))
    if img.width > MAX_W:
        img = img.resize((MAX_W, max(1, round(img.height * MAX_W / img.width))), Image.LANCZOS)
    img = img.quantize(colors=64)  # text and line art: much smaller PNG, still sharp
    out = io.BytesIO()
    img.save(out, "PNG", optimize=True)
    img.close()
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def snapshot_questions(doc, pages):
    """doc, pages (text of each page) -> (questions, notes). Questions carry the picture in 'images' and snap=True."""
    pg = Pages(doc)
    n = len(doc)
    scanned = {i for i, t in enumerate(pages) if len(t.strip()) < 30}
    text_pages = [i for i in range(n) if i not in scanned]
    starts = _find_starts(doc, pg, text_pages) if text_pages else []
    notes, specs = [], []
    if len(starts) >= 2:
        specs = [_start_spec(doc, pg, starts, i, scanned) for i in range(len(starts))]
        gap_pages = sorted(scanned)
    else:
        gap_pages = list(range(n))
        notes.append("Question numbers were not found, so every block of the page was cut out as its own picture. "
                     "Remove the pieces that are not questions (headings, instructions).")
    for p in gap_pages:
        for top, bot in _gap_bands(doc[p]):
            specs.append({"p": p, "y": top, "num": None, "pieces": [(p, top, bot, [])], "ans": None, "k": None})
    if scanned and len(starts) >= 2 and gap_pages:
        notes.append(f"{len(scanned)} scanned page(s) were cut at their blank gaps: check that each picture is one question.")
    specs.sort(key=lambda s: (s["p"], s["y"]))

    counts = Counter(s["k"] for s in specs if s["k"])
    usual = max(counts, key=lambda c: (counts[c], c)) if counts else None
    raw = [l.strip() for t in pages for l in t.splitlines() if l.strip()]
    key, _, ktypes = _split_key(raw)
    nums = [s["num"] for s in specs if s["num"]]
    key_ok = bool(key) and len(nums) == len(set(nums))  # numbers that repeat (chapters) cannot be matched to a key

    out, failed = [], 0
    for s in specs[:MAX_Q]:
        try:
            uri = _picture(doc, pg, s["pieces"])
        except Exception:
            log.exception("Picture cut failed on page %s", s["p"] + 1)
            uri, failed = None, failed + 1
        if not uri:
            continue
        num = s["num"]
        a = s["ans"] if s["ans"] is not None else (key.get(num) if key_ok and num else None)
        kt = ktypes.get(num) if key_ok and num else None
        k = s["k"] or usual or 4
        nat = kt == "NAT" or (isinstance(a, dict) and not counts and kt is None)
        if nat:
            q = {"type": "NAT", "options": {}, "answer": a if isinstance(a, dict) else None}
        else:
            opts = {chr(65 + i): "(see picture)" for i in range(k)}
            if isinstance(a, dict) and a["min"] == a["max"] and a["min"] == int(a["min"]) and 1 <= a["min"] <= k:
                a = [chr(64 + int(a["min"]))]
            letters = [x for x in (a if isinstance(a, list) else []) if x in opts]
            multi = len(letters) > 1 or (kt == "MSQ" and bool(letters))
            q = {"type": "MSQ" if multi else "MCQ", "options": opts,
                 "answer": letters if multi else (letters[0] if letters else None)}
        q.update(id=num or len(out) + 1, question="(see picture)", explanation="", images=[uri], snap=True)
        out.append(q)
    pg.clear()
    if failed:
        notes.append(f"{failed} piece(s) could not be cut and were left out.")
    if len(specs) > MAX_Q:
        notes.append(f"Only the first {MAX_Q} questions were cut.")
    if out and not any(q["answer"] for q in out):
        notes.append("No answers were found in the PDF. Tick the correct option under each picture.")
    if out and not counts:
        notes.append("Options could not be counted from the text, so 4 options (A-D) were assumed. Change this per question with + Add option.")
    return out, notes