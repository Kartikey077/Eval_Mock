"""Rule-based parser: plain text -> MCQ / MSQ / NAT questions."""
import re
from collections import Counter

NUM = r"-?\d+(?:\.\d+)?"
LET = r"\(?[A-Ea-e]\)?"
LETTERS = rf"{LET}(?:\s*(?:,|;|&|/|and)?\s*{LET})*"
VALUE = rf"(?:{LETTERS}(?!\w)|{NUM}(?:\s*(?:to|[–—-])\s*{NUM})?)"

# "Q.1 text", "Q. 4 text", "Q1. text", "1. text", "1) text"
RE_Q = re.compile(r"^\s*(?:Q(?:uestion)?\s*\.?\s*(\d+)\s*[\.\):\-]?|(\d+)\s*[\.\):\-])\s+(.+)", re.I)
RE_QALONE = re.compile(r"^\s*Q(?:uestion)?\s*\.?\s*(\d+)\s*[\.\):]?\s*$", re.I)  # "Q.46" alone, text on next line
RE_RANGE = re.compile(r"^\s*Q\.?\s*\d+\s*[–—-]\s*Q\.?\s*\d+", re.I)  # "Q.6 – Q.10 Carry TWO marks Each"
RE_OPT = re.compile(r"^\s*\(?([A-Ea-e])[\.\)]\s+(.+)")
RE_ANS = re.compile(r"^\s*(?:correct\s+(?:answers?|options?)|answers?|ans\.?)\s*[:\-.]?\s*(.+)", re.I)
RE_EXP = re.compile(r"^\s*(?:explanation|solution|reason)\s*[:\-]\s*(.*)", re.I)
RE_KEY = re.compile(r"^\s*(?:answer\s*key|answers|solutions?)\s*:?\s*(.*)$", re.I)
SEP = r"(?:[\.\):\-–—=→]\s*|\s+)"
RE_PAIR = re.compile(rf"(?:Q\.?\s*)?(\d+)\s*{SEP}({VALUE})", re.I)
PAIRS_ONLY = re.compile(rf"^(?:\s*(?:Q\.?\s*)?\d+\s*{SEP}{VALUE}\s*[,;]?)+\s*$", re.I)
# official key tables:  "27 2 MSQ CS-2 B;C 1"   (no. session type section key marks)
# wording that usually means "more than one option may be correct" (a hint only, never certain)
RE_MULTI = re.compile(r"\bis\s*/\s*are\b|\bare\s*/\s*is\b|\w\(s\)|\bone or more\b|\bselect all\b|\bmore than one\b", re.I)
# "Question Number : 35   Correct : 2   Wrong : -0.66"  (question marker used by some papers)
RE_QNUM = re.compile(r"^\s*Question\s*Number\s*:\s*(\d[\d ]*?)\s*(?:Correct\b|$)", re.I)
# an option whose "text" is only other option labels, e.g. "(B)" - happens when options are pictures
RE_LABELONLY = re.compile(r"^(?:(?:\([A-Ea-e]\)|[A-Ea-e][\.\)])\s*)+$")
def junk_option(t):
    """True for an 'option' that is really text taken from a picture: only labels like '(B)', or a longer text
    that contains another option label such as '(B) WR X WR N Garden ...'."""
    t = t.strip()
    return bool(RE_LABELONLY.match(t) or re.search(r"\(\s*[A-Ea-e]\s*\)", t))


RE_TAILLABELS = re.compile(r"(?:\s*\([A-Ea-e]\))+\s*$")
RE_ROW = re.compile(r"(?<!\d)(\d{1,3})\s+\d+\s+(MCQ|MSQ|NAT)\s+\S+\s+(.+?)\s+\d+(?:\.\d+)?\s*$")


def clean_pages(pages):
    """Join page texts, dropping headers/footers that repeat on many pages (e.g. 'Page 3 of 42')."""
    if len(pages) < 4:
        return "\n".join(pages)

    def norm(l):
        return re.sub(r"\d+", "#", re.sub(r"\s+", " ", l.strip().lower()))

    def candidate(l):
        return len(l.strip()) >= 15 and not (RE_OPT.match(l) or RE_Q.match(l) or RE_QNUM.match(l) or RE_ANS.match(l))

    cnt = Counter()
    for p in pages:
        for n in {norm(x) for x in p.splitlines() if candidate(x)}:
            cnt[n] += 1
    bad = {n for n, c in cnt.items() if c >= max(3, len(pages) // 2)}
    return "\n".join(x for p in pages for x in p.splitlines() if norm(x) not in bad)


def parse_answer(raw):
    """'B' -> ['B'] | 'A, C' -> ['A','C'] | '5.2' or '4.9 to 5.1' -> {'min','max'} | None"""
    raw = re.sub(r"^(?:option|opt\.?)\s*", "", raw.strip().rstrip(".").strip(), flags=re.I)
    if re.fullmatch(LETTERS, raw):
        return sorted(set(re.findall(r"[A-E]", raw.upper())))
    m = re.fullmatch(rf"({NUM})(?:\s*(?:to|[–—-])\s*({NUM}))?(?:\s+[^\d\s]\S*)?", raw)
    if m:
        lo = float(m.group(1)); hi = float(m.group(2) or lo)
        return {"min": min(lo, hi), "max": max(lo, hi)}
    return None


def _finalize(q, a, kt=None):
    q["question"] = q["question"].strip()
    if q["options"]:
        letters = a if isinstance(a, list) else []
        multi = len(letters) > 1 or (kt == "MSQ" and bool(letters))
        q["type"] = "MSQ" if multi else "MCQ"
        q["answer"] = letters if multi else (letters[0] if letters else None)
        if not letters and kt is None and RE_MULTI.search(q["question"]):
            q["hint"] = True  # wording suggests several answers: only a hint, the type stays MCQ until a key/AI/you decide
    else:
        q["type"] = "NAT"
        q["answer"] = a if isinstance(a, dict) else None
    return q


def _table_key(lines):
    """Official answer-key table (Q.No / Session / Type / Section / Key / Marks) at the end of the PDF."""
    key, types, idx = {}, {}, []
    for i, l in enumerate(lines):
        l = re.sub(r"^Page\s+\d+\s+of\s+\d(?=\d+\s+\d+\s+(?:MCQ|MSQ|NAT)\b)", "", l)  # "Page 1 of 231 2 MSQ.." glue
        m = RE_ROW.search(l)
        a = parse_answer(m.group(3)) if m else None
        if a is not None:
            n = int(m.group(1)); key[n], types[n] = a, m.group(2).upper(); idx.append(i)
    if len(key) < 3:
        return None
    start = idx[0]
    for j in range(max(0, start - 6), start):  # cut at the "Answer Key for ..." title just above the table
        if re.match(r"\s*answer\s*key\b", lines[j], re.I):
            start = j; break
    return key, lines[:start], types


def _split_key(lines):
    """Pull answer-key blocks out of the text: tables, 'Answer Key: 1-B 2-C', page-end keys,
    or a heading-less block of '1. B' lines at the end."""
    t = _table_key(lines)
    if t:
        return t[0], t[1], t[2]
    key, keep, i = {}, [], 0

    def take(s):
        for n, v in RE_PAIR.findall(s):
            a = parse_answer(v)
            if a is not None:
                key[int(n)] = a

    while i < len(lines):
        m = RE_KEY.match(lines[i])
        rest = m.group(1).strip() if m else ""
        if m and (not rest or len(RE_PAIR.findall(rest)) >= 2):
            take(rest); i += 1
            while i < len(lines) and PAIRS_ONLY.match(lines[i]):
                take(lines[i]); i += 1
            continue
        keep.append(lines[i]); i += 1
    if not key:
        j = len(keep)
        while j > 0 and PAIRS_ONLY.match(keep[j - 1]):
            j -= 1
        if len(keep) - j >= 3:
            for l in keep[j:]:
                take(l)
            keep = keep[:j]
    return key, keep, {}


def _parse_lines(lines):
    qs, cur, last = [], None, None
    for ln in lines:
        if RE_RANGE.match(ln):  # section heading like "Q.6 – Q.10 Carry TWO marks Each"
            last = None
            continue
        m = RE_QNUM.match(ln)
        if m:
            if cur and cur["question"].strip():
                qs.append(cur)
            cur = {"id": int(re.sub(r"\s+", "", m.group(1))), "type": "MCQ", "question": "",
                   "options": {}, "answer": None, "explanation": ""}
            last = "q"
            continue
        m = RE_ANS.match(ln)
        if m and cur is not None:
            a = parse_answer(m.group(1))
            if a is not None:
                cur["_a"], last = a, None
                continue
        m = RE_EXP.match(ln)
        if m and cur is not None:
            cur["explanation"], last = m.group(1).strip(), "exp"
            continue
        parts = re.split(r"\s+(?=\(?[A-Ea-e][\.\)]\s)", ln)  # several options on one line
        ms = [RE_OPT.match(p) for p in parts]
        if cur is not None and all(ms):
            for mm in ms:
                last = mm.group(1).upper()
                cur["options"][last] = mm.group(2).strip()
            continue
        m = RE_Q.match(ln)
        alone = None if m else RE_QALONE.match(ln)
        if m or alone:
            num = int((m.group(1) or m.group(2)) if m else alone.group(1))
            if alone or cur is None or cur["options"] or "_a" in cur or num == cur["id"] + 1:
                if cur and cur["question"].strip():
                    qs.append(cur)
                cur = {"id": num, "type": "MCQ", "question": m.group(3).strip() if m else "",
                       "options": {}, "answer": None, "explanation": ""}
                last = "q"
                continue
        if cur is not None and last:  # wrapped line
            if last == "q":
                cur["question"] += " " + ln
            elif last == "exp":
                cur["explanation"] += " " + ln
            else:
                cur["options"][last] += " " + ln
    if cur and cur["question"].strip():
        qs.append(cur)
    return qs


def parse_with_info(text, picture_opts=None):
    """Returns (questions, info). info says whether an answer key was found, applied, or rejected.
    picture_opts = {position in list: ['A','B','C','D']} for questions whose options are pictures: those options
    get the text "(see figure)" so the question, the answer key and the test all work normally."""
    key, lines, ktypes = _split_key([l.strip() for l in text.splitlines() if l.strip()])
    qs = _parse_lines(lines)
    for pos, q in enumerate(qs):
        letters = (picture_opts or {}).get(pos)
        if letters:
            q["options"] = {L: t for L, t in q["options"].items() if not junk_option(t)}
            for L in letters:
                q["options"].setdefault(L, "(see figure)")
            q["options"] = dict(sorted(q["options"].items()))
            q["question"] = RE_TAILLABELS.sub("", q["question"])
    info = {"key_found": bool(key), "key_applied": False, "mismatch": []}
    if key:
        byid = {q["id"]: q for q in qs}
        for n, a in key.items():
            q = byid.get(n)
            if not q:
                continue
            kt, has = ktypes.get(n), bool(q["options"])
            if ((kt == "NAT" and has) or (kt in ("MCQ", "MSQ") and not has)
                    or (isinstance(a, list) and (not has or any(x not in q["options"] for x in a)))
                    or (isinstance(a, dict) and has)):
                info["mismatch"].append(n)
        if len(info["mismatch"]) >= 3:  # key clearly belongs to another paper: never apply it
            key, ktypes = {}, {}
        else:
            info["mismatch"], info["key_applied"] = [], True
    out = []
    for q in qs:
        a = q.pop("_a", None)
        out.append(_finalize(q, key.get(q["id"]) if a is None else a, ktypes.get(q["id"])))
    return out, info


def parse_questions(text):
    return parse_with_info(text)[0]


def complete(q):
    if q["type"] == "NAT":
        return isinstance(q["answer"], dict)
    return len(q["options"]) >= 2 and bool(q["answer"])


def structured(q):
    return len(q["options"]) >= 2 or (q["type"] == "NAT" and len(q["question"]) >= 20)


def quality_ok(qs, strict=False):
    """True when the questions are usable. strict=True (OCR text) allows no incomplete question.
    A paper with no answers at all is judged on structure only (answers can be added later)."""
    if not qs:
        return False
    ids = {q["id"] for q in qs}
    if len(qs) >= 5 and len(ids) < 0.9 * (max(ids) - min(ids) + 1):  # numbering has gaps: questions were lost
        return False
    check = structured if all(not q["answer"] for q in qs) else complete
    return sum(map(check, qs)) / len(qs) >= (1.0 if strict else 0.9)

def problems(qs):
    """Return (incomplete, few_options) as 1-based POSITIONS in the list (the 'Question N' you see on screen).
    Paper numbers can repeat, so positions are the only unambiguous label."""
    incomplete, few = [], []
    no_answers = all(not q["answer"] for q in qs)  # paper has no key: judge structure only, as quality_ok does
    counts = Counter(len(q["options"]) for q in qs if q["type"] != "NAT" and len(q["options"]) >= 2)
    usual = max(counts, key=lambda n: (counts[n], n)) if counts else 4  # the paper's normal option count (4, or 3, ...)
    for pos, q in enumerate(qs, 1):
        if not (structured(q) if no_answers else complete(q)):
            incomplete.append(pos)
        elif q["type"] != "NAT" and 2 <= len(q["options"]) < usual and not q.get("images"):
            few.append(pos)
    return incomplete, few


def needs_ai(qs, strict=False):
    """True when AI re-reading is worth trying (broken questions, or quality below threshold)."""
    if not quality_ok(qs, strict):
        return True
    incomplete, few = problems(qs)
    return bool(incomplete or few)


def rows_text(page, tol=3):
    """Rebuild text row by row from word positions. Fixes tables where every cell ('(A)' | 'Accelerate')
    is otherwise returned on its own line."""
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if rows and abs(rows[-1][0] - yc) <= tol:
            rows[-1][1].append(w)
        else:
            rows.append([yc, [w]])
    lines = [[yc, " ".join(w[4] for w in sorted(ws, key=lambda w: w[0]))] for yc, ws in rows]
    out = []
    for k, (yc, t) in enumerate(lines):
        if re.fullmatch(r"Q\.?\s*\d+", t.strip()):  # a lone "Q.46" label sitting a few points off its text line
            dprev = yc - out[-1][0] if out else 99
            dnext = lines[k + 1][0] - yc if k + 1 < len(lines) else 99
            if min(dprev, dnext) <= 12:
                if dprev <= dnext:
                    out[-1][1] = t.strip() + " " + out[-1][1]
                else:
                    lines[k + 1][1] = t.strip() + " " + lines[k + 1][1]
                continue
        out.append([yc, t])
    return "\n".join(t for _, t in out)


def has_options(text):
    return len(re.findall(r"(?m)^\s*\(?[A-Ea-e][\.\)]\s+\S", text)) >= 4


def image_based(page, text):
    """True when a page is a picture (scan/screenshot), even if a few header words exist as real text."""
    t = text.strip()
    if len(t) < 30:
        return True
    if has_options(t) or re.search(r"\(\s*[A-Ea-e]\s*\)\s*\S", t):
        return False
    return len(re.sub(r"\s+", " ", t)) < 300 and bool(page.get_images())  # a picture with only a few header words of real text


def _score(qs):
    return sum(2 if complete(q) else (1 if len(q["options"]) >= 2 else 0) for q in qs)


def best_text_parse(doc, pages, ocr_idx=()):
    """Read the PDF two ways - normal text order, and row by row (fixes table layouts) - and keep
    whichever version gives more usable questions."""
    text = clean_pages(pages)
    qs, info = parse_with_info(text) if text.strip() else ([], {})
    rows = [pages[i] if i in ocr_idx else rows_text(doc[i]) for i in range(len(pages))]
    t2 = clean_pages(rows)
    q2, i2 = parse_with_info(t2) if t2.strip() else ([], {})
    return (t2, q2, i2) if _score(q2) > _score(qs) else (text, qs, info)