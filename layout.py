"""Readable layout for questions that contain a table, program code or a list.

Plain text extraction flattens everything into one long line ("Process P1 P2 P3 P4 Arrival time 0 1 3 4 ...").
Here the page is read by position instead, and the question becomes a list of blocks:
    {"t": "p",     "x": "paragraph text"}                      (line breaks kept as in the PDF)
    {"t": "code",  "x": "program text, indentation kept"}
    {"t": "table", "caption": "Student" or None, "rows": [["Roll no.", "Name"], ["1", "Amit"], ...]}
Only questions that really contain a table, code or a list get blocks; all others stay plain text.
The options are NOT part of the blocks (they are shown separately).
"""
import bisect
import re

import pymupdf

from figures import RE_ANSLINE, RE_OPTLINE, RE_SOLLINE, Pages, question_bands

RE_SHEETCELL = re.compile(r"^(?:Q\.?\d+\.?|\(?[A-Ea-e]\)?\.?)?$")  # first column of a question sheet: Q.5, (A), empty


def _is_sheet_table(tab):
    """A table whose first column holds only question numbers and option labels is the PAGE LAYOUT of a
    question-sheet PDF (every question in a bordered table), not a table that belongs to a question."""
    try:
        firsts = [re.sub(r"\s+", "", (r[0] or "")) for r in tab.extract() if r]
    except Exception:
        return False
    full = [f for f in firsts if f]
    return len(full) >= 2 and all(RE_SHEETCELL.match(f) for f in full)


def _detect_sheet(doc, pg, tabs):
    """True when the PDF is a question sheet. Looks at the first few pages that have ruled tables."""
    seen = 0
    for p in range(len(doc)):
        found = _tables_on(doc[p], pg.drawings(p))
        if not found:
            continue
        tabs[p] = found
        seen += 1
        if any(_is_sheet_table(t) for t in found):
            return True
        if seen >= 3:
            break
    return False


RE_QPREFIX = re.compile(r"^\s*(?:Question\s*Number\s*:\s*\d+|Q(?:uestion)?\s*\.?\s*\d+\s*[\.\):\-]?)\s*", re.I)
RE_LISTITEM = re.compile(r"^\s*(?:[IVX]{1,4}|S\d{1,2})[\.\)]\s+\S")  # "I. text", "II. text", "S1. text"
RE_LISTLABEL = re.compile(r"^(?:[IVX]{1,4}|S\d{1,2})[\.\)]$")  # a lone "I." or "S1."
RE_QALONE = re.compile(r"^\s*Q(?:uestion)?\s*\.?\s*\d+\s*[\.\):]?\s*$", re.I)  # the question label on its own

CODE_LINE = re.compile(r"""^(?:
    \#\s*(?:include|define|ifdef|ifndef|endif)\b |
    (?:int|float|char|void|double|long|short|unsigned|signed|struct|static|const|bool)\b.*[;{)]\s*$ |
    (?:for|while|if|switch)\s*\(.*$ |
    (?:else|do)\b\s*\{?\s*$ |
    return\b.*;\s*$ |
    (?:printf|scanf|fork|exit|malloc|free)\s*\(.*\)\s*;?\s*$ |
    [{}]\s*;?\s*$ |
    [A-Za-z_]\w*(?:\[[^\]]*\])*\s*(?:[-+*/%&|^]?=|\+\+|--)(?!=).*;\s*$ |
    [A-Za-z_]\w*\s*\([^()]*\)\s*;\s*$ |
    (?:SELECT|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|JOIN|INSERT|UPDATE|DELETE)\b.*
    )""", re.X)
CODE_HINT = re.compile(r"[;{}#]|^\s*SELECT\b", re.M)


# ---------------------------------------------------------------- tables

def _tables_on(page, drawings):
    """Tables found on a page. Table detection is slow, so it only runs where ruling lines form a grid
    (several horizontal AND vertical lines), and only over the area those lines cover."""
    H = page.rect.height
    hor, ver = [], []
    for d in drawings:
        r = d["rect"]
        if r.y1 < 0.06 * H or r.y0 > 0.935 * H:  # running header / footer rules
            continue
        if r.width > 20 and r.height < 2.5:
            hor.append(r)
        elif r.height > 12 and r.width < 2.5:
            ver.append(r)
    if len(hor) < 3 or len(ver) < 3:
        return []
    area = pymupdf.Rect(hor[0])
    for r in hor + ver:
        area.include_rect(r)
    area = area + (-6, -6, 6, 6)
    try:
        return page.find_tables(clip=area).tables  # needs PyMuPDF 1.23 or newer
    except Exception:
        return []


def _edges(vals, tol=2.5):
    out = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [sum(g) / len(g) for g in out]


def _cell_text(ws):
    """Words of one cell -> text. Words on (nearly) the same height form a line; lines are joined with \\n."""
    rows = []
    for w in sorted(ws, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        cy = (w[1] + w[3]) / 2
        if rows and abs(rows[-1][0] - cy) <= 4:  # 4pt: subscripts and superscripts stay on their line
            rows[-1][1].append(w)
        else:
            rows.append([cy, [w]])
    return "\n".join(" ".join(x[4] for x in sorted(r[1], key=lambda w: w[0])) for r in rows).strip()


def _table_blocks(words, tab):
    """Rebuild a detected table from word positions (the library's own cell text drops spaces).
    Returns a list of table blocks (a table with an empty spacer column is really two tables), or None."""
    cells = [c for c in tab.cells if c]
    xs = _edges([c[0] for c in cells] + [c[2] for c in cells])
    ys = _edges([c[1] for c in cells] + [c[3] for c in cells])
    if len(xs) < 3 or len(ys) < 3:
        return None
    nc, nr = len(xs) - 1, len(ys) - 1
    cw = [[[] for _ in range(nc)] for _ in range(nr)]
    for w in words:
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        if xs[0] <= cx <= xs[-1] and ys[0] <= cy <= ys[-1]:
            cw[min(nr - 1, bisect.bisect_right(ys, cy) - 1)][min(nc - 1, bisect.bisect_right(xs, cx) - 1)].append(w)
    grid = [[_cell_text(cw[r][c]) for c in range(nc)] for r in range(nr)]
    if sum(1 for row in grid for t in row if t) < 4:
        return None

    # split into separate tables at WIDE empty columns (a gap between two tables); thin empty columns are
    # just border artefacts and are dropped further down
    gap = [all(not grid[r][c] for r in range(nr)) and xs[c + 1] - xs[c] > 15 for c in range(nc)]
    groups, cur = [], []
    for c in range(nc):
        if gap[c]:
            if cur:
                groups.append(cur)
            cur = []
        else:
            cur.append(c)
    if cur:
        groups.append(cur)

    blocks = []
    for cols in groups:
        rows = [[grid[r][c] for c in cols] for r in range(nr)]
        rows = [r for r in rows if any(r)]
        caption = None
        if len(rows) >= 2 and sum(1 for t in rows[0] if t) == 1 and len(rows[0]) > 1:
            caption = next(t for t in rows[0] if t)  # a one-cell first row is the table's title ("Student", "P")
            rows = rows[1:]
        keep = [i for i in range(len(rows[0])) if any(r[i] for r in rows)] if rows else []
        rows = [[r[i] for i in keep] for r in rows]
        if rows and keep:
            blocks.append({"t": "table", "caption": caption, "rows": rows})
    return blocks or None


# ---------------------------------------------------------------- text blocks

def _merge_rows(lines):
    """Pieces that sit on the same row ("I." + its sentence, f1 = ..., f2 = ..., f3 = ...) become one line.
    Same row = the vertical centres are close (labels and superscripts sit a few points off the baseline).
    Each row keeps its cells: pieces less than 10pt apart form one cell."""
    ls = sorted(lines, key=lambda l: (l.get("p", 0), (l["y0"] + l["y1"]) / 2))
    rows = []
    for l in ls:
        cy, h = (l["y0"] + l["y1"]) / 2, l["y1"] - l["y0"]
        if rows and rows[-1]["p"] == l.get("p", 0) and abs(rows[-1]["cy"] - cy) <= 0.5 * min(rows[-1]["h"], h) + 1:
            rows[-1]["items"].append(l)
        else:
            rows.append({"p": l.get("p", 0), "cy": cy, "h": h, "items": [l]})
    out = []
    for r in rows:
        items = sorted(r["items"], key=lambda l: l["x0"])
        cells = []
        for l in items:
            if cells and (l["x0"] - cells[-1][1] < 10 or RE_LISTLABEL.match(cells[-1][2])):
                c = cells[-1]
                cells[-1] = (c[0], max(c[1], l["x1"]), c[2] + " " + l["text"])
            else:
                cells.append((l["x0"], l["x1"], l["text"]))
        out.append({"x0": items[0]["x0"], "y0": min(l["y0"] for l in items), "x1": max(l["x1"] for l in items),
                    "y1": max(l["y1"] for l in items), "p": r["p"], "cells": cells,
                    "mono": all(l.get("mono") for l in items),
                    "text": "    ".join(c[2] for c in cells)})
    return out


RE_MATHGLYPH = re.compile(r"[\uf000-\uf8ff\u23a0-\u23af\u2320\u2321]")  # symbol-font glyphs, big-brace pieces
RE_LABELCELL = re.compile(r"^(?:\(?[A-Ea-e]\)?[\.\)]?|[IVX]{1,4}[\.\)]|S\d{1,2}[\.\)])$")  # "(A)", "I.", "S1."


def _grid_table(run):
    """Consecutive rows with 2+ cells that line up in columns -> a table block (text-aligned table, no ruling
    lines needed). Returns (block, rows_used) or None. The longest valid prefix of the run is used."""
    for end in range(len(run), 2, -1):  # a table needs at least 3 rows
        sub = run[:end]
        iv = sorted((c[0], c[1]) for r in sub for c in r["cells"])
        cols = []
        for a, b in iv:  # columns = overlapping x-ranges of the pieces
            if cols and a <= cols[-1][1] + 6:
                cols[-1][1] = max(cols[-1][1], b)
            else:
                cols.append([a, b])
        if len(cols) < 2:
            continue

        def col_of(c):
            mid = (c[0] + c[1]) / 2
            return next((i for i, (a, b) in enumerate(cols) if a - 1 <= mid <= b + 1), None)

        grid, ok = [], True
        for r in sub:
            idx = [col_of(c) for c in r["cells"]]
            if None in idx or len(set(idx)) != len(idx):
                ok = False
                break
            row = [""] * len(cols)
            for i, c in zip(idx, r["cells"]):
                row[i] = c[2]
            grid.append(row)
        if not ok:
            continue
        firsts = [row[0] for row in grid if row[0]]
        if firsts and all(RE_LABELCELL.match(t) for t in firsts):
            return None  # a list of labelled items ("I. ...", "(A) ..."), not a table
        if any(RE_MATHGLYPH.search(t) for row in grid for t in row):
            return None  # brace / symbol-font pieces: a formula laid out in rows, not a table
        keep = [i for i in range(len(cols)) if any(row[i] for row in grid)]
        grid = [[row[i] for i in keep] for row in grid]
        return {"t": "table", "caption": None, "rows": grid}, end
    return None


def _split_tables(rows):
    """Replace runs of column-aligned rows by table blocks. Returns a list of ('rows', [row..]) / ('table', block)."""
    parts, buf, i, n = [], [], 0, len(rows)
    while i < n:
        if len(rows[i]["cells"]) >= 2:
            j = i + 1
            while (j < n and len(rows[j]["cells"]) >= 2
                   and rows[j]["y0"] - rows[j - 1]["y1"] < 1.2 * (rows[j - 1]["y1"] - rows[j - 1]["y0"]) + 10):
                j += 1
            got = _grid_table(rows[i:j]) if j - i >= 3 else None
            if got:
                if buf:
                    parts.append(("rows", buf))
                    buf = []
                parts.append(("table", got[0]))
                i += got[1]
                continue
        buf.append(rows[i])
        i += 1
    if buf:
        parts.append(("rows", buf))
    return parts


def _text_blocks(lines, margin):
    """Lines (no tables) -> paragraph and code blocks."""
    if not lines:
        return []
    flags = [bool(l.get("mono")) or bool(CODE_LINE.match(l["text"].strip())) for l in lines]
    code = [False] * len(lines)
    i = 0
    while i < len(lines):  # a run of code-looking lines counts as code when it is long enough and looks like code
        if flags[i]:
            j = i
            while j < len(lines) and flags[j]:
                j += 1
            run = "\n".join(l["text"] for l in lines[i:j])
            if j - i >= 2 and (CODE_HINT.search(run) or all(l.get("mono") for l in lines[i:j])):
                for k in range(i, j):
                    code[k] = True
            i = j
        else:
            i += 1

    blocks, i = [], 0
    while i < len(lines):
        j = i
        while j < len(lines) and code[j] == code[i]:
            j += 1
        seg = lines[i:j]
        if code[i]:
            levels = _edges([l["x0"] for l in seg], tol=3)  # distinct indentation positions, left to right
            text = "\n".join("    " * min(range(len(levels)), key=lambda i: abs(levels[i] - l["x0"])) + l["text"]
                             for l in seg)
            blocks.append({"t": "code", "x": text})
        else:
            cur, parts = None, []
            for k, l in enumerate(seg):
                if cur is not None:
                    prev = seg[k - 1]
                    gap = l["y0"] - prev["y1"]
                    if gap > 0.9 * (prev["y1"] - prev["y0"]):  # a blank line: new paragraph
                        parts.append(cur)
                        cur = None
                if cur is None:
                    cur = l["text"]
                else:
                    wrapped = seg[k - 1]["x1"] >= margin - 22 and not RE_LISTITEM.match(l["text"])  # full line = wrapped
                    cur += (" " if wrapped else "\n") + l["text"]
            if cur is not None:
                parts.append(cur)
            blocks += [{"t": "p", "x": p} for p in parts if p.strip()]
        i = j
    return blocks


def find_layout(doc, qs, skip=(), starts_bands=None, images=None):
    """Returns {position: [blocks]} for questions that contain a table, code or a list.
    Questions that have a figure are left alone (their layout is the picture)."""
    starts, bands = starts_bands or question_bands(doc, qs, skip)
    pg = Pages(doc)
    images = images or {}
    tabs = {}
    out = {}
    sheet = _detect_sheet(doc, pg, tabs)  # question-sheet PDF: its big table is page layout, not content
    if sheet:
        tabs.clear()
    for k in range(len(qs)):
        if k not in bands or k in images:
            continue
        items, margin_lines, first_text = [], [], True
        has_table = False
        for bi, (p, top, bot) in enumerate(bands[k]):
            page = doc[p]
            if p not in tabs:
                tabs[p] = [] if sheet else _tables_on(page, pg.drawings(p))
            H = page.rect.height
            lines = [l for l in pg.lines(p, top, bot)  # no running header / footer, no loose "Q.5" label
                     if l["y1"] > 0.06 * H and l["y0"] < 0.935 * H and not RE_QALONE.match(l["text"])]
            cut = next((l["y0"] for l in lines if RE_OPTLINE.match(l["text"])
                        or RE_ANSLINE.match(l["text"]) or RE_SOLLINE.match(l["text"])), None)
            if cut is not None:
                lines = [l for l in lines if l["y1"] <= cut + 1]
                bot = min(bot, cut)
            boxes = []
            for tab in tabs[p]:
                bx = tab.bbox
                if top <= (bx[1] + bx[3]) / 2 <= bot:
                    blocks = _table_blocks(pg.words(p), tab)
                    if blocks:
                        has_table = True
                        boxes.append(bx)
                        items.append((p, bx[1], "table", blocks))
            for l in lines:
                cy = (l["y0"] + l["y1"]) / 2
                if not any(b[0] - 2 <= (l["x0"] + l["x1"]) / 2 <= b[2] + 2 and b[1] - 2 <= cy <= b[3] + 2 for b in boxes):
                    items.append((p, l["y0"], "line", dict(l, p=p, mono=pg.is_mono(p, l))))
                    margin_lines.append(l)
            if cut is not None:
                break
        if not items:
            continue
        items.sort(key=lambda it: (it[0], it[1]))
        full = [l for l in margin_lines if len(l["text"]) > 60]
        margin = max((l["x1"] for l in full), default=max((l["x1"] for l in margin_lines), default=0))

        blocks, seg = [], []

        def flush(seg):
            out = []
            for kind, payload in _split_tables(_merge_rows(seg)):
                if kind == "table":
                    out.append(payload)
                else:
                    out += _text_blocks(payload, margin)
            return out

        for _p, _y, kind, payload in items:
            if kind == "line":
                seg.append(payload)
            else:
                blocks += flush(seg)
                seg = []
                blocks += payload
        blocks += flush(seg)

        # drop the "Q.12" label in front of the first paragraph
        for b in blocks:
            if b["t"] == "p":
                b["x"] = RE_QPREFIX.sub("", b["x"], count=1).strip()
                break
        blocks = [b for b in blocks if b["t"] != "p" or b["x"]]

        has_table = has_table or any(b["t"] == "table" for b in blocks)
        has_code = any(b["t"] == "code" for b in blocks)
        n_items = sum(1 for l in _merge_rows(margin_lines) if RE_LISTITEM.match(l["text"]))
        if has_table or has_code or n_items >= 2:
            out[k] = blocks
    return out