"""Rule-based parser: plain text -> MCQ / MSQ / NAT questions."""
import os
import re
from collections import Counter

NUM = r"-?\d+(?:\.\d+)?"
LET = r"\(?[A-Ea-e]\)?"
LETTERS = rf"{LET}(?:\s*(?:,|;|&|/|and)?\s*{LET})*"
VALUE = rf"(?:{LETTERS}(?!\w)|{NUM}(?:\s*(?:to|[–—-])\s*{NUM})?)"

# a question starts ONLY with "Q.1 text", "Q. 4 text", "Q1. text", "Question 5 text" (never with a bare "1." / "1)")
RE_Q = re.compile(r"^\s*Q(?:uestion)?\s*\.?\s*(\d+)\s*[\.\):\-]?\s+(.+)", re.I)
RE_QALONE = re.compile(r"^\s*Q(?:uestion)?\s*\.?\s*(\d+)\s*[\.\):]?\s*$", re.I)  # "Q.46" alone, text on next line
RE_RANGE = re.compile(r"^\s*Q\.?\s*\d+\s*[–—-]\s*Q\.?\s*\d+", re.I)  # "Q.6 – Q.10 Carry TWO marks Each"
RE_OPT = re.compile(r"^\s*(?:\(?([A-Ea-e])[\.\)]|\(([1-5])\))\s+(.+)")  # (a) a. A) and (1) (2): digits become A B C


def opt_match(p):
    """(letter, text) when p starts an option, else None. '(1)' is read as option A, '(2)' as B ..."""
    m = RE_OPT.match(p)
    if not m:
        return None
    return (m.group(1).upper() if m.group(1) else chr(64 + int(m.group(2)))), m.group(3).strip()


RE_ANS = re.compile(r"^\s*(?:correct\s+(?:answers?|options?)|answers?|ans\.?)\s*[:\-.]?\s*(.+)", re.I)
RE_EXP = re.compile(r"^\s*(?:explanation|solution|reason)\s*[:\-]\s*(.*)", re.I)
RE_KEY = re.compile(r"^\s*(?:answer\s*key|answers|solutions?)\s*:?\s*(.*)$", re.I)
SEP = r"(?:[\.\):\-–—=→]\s*|\s+)"
RE_PAIR = re.compile(rf"(?:Q\.?\s*)?(\d+)\s*{SEP}({VALUE})", re.I)
PAIRS_ONLY = re.compile(rf"^(?:\s*(?:Q\.?\s*)?\d+\s*{SEP}{VALUE}\s*[,;]?)+\s*$", re.I)
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
# official key tables:  "27 2 MSQ CS-2 B;C 1"   (no. session type section key marks)
RE_ROW = re.compile(r"(?<!\d)(\d{1,3})\s+\d+\s+(MCQ|MSQ|NAT)\s+\S+\s+(.+?)\s+\d+(?:\.\d+)?\s*$")


def option_parts(ln):
    """[(letter, text)] for the options on one line, or None. Several options on one line ('(a) x (b) y') are split
    only when their letters run in order (a, b, c...). So '(d) Both (a) and (b) are correct' stays ONE option."""
    parts = re.split(r"\s+(?=\(?[A-Ea-e][\.\)]\s|\([1-5]\)\s)", ln)
    ms = [opt_match(p) for p in parts]
    if all(ms):
        ls = [ord(m[0]) for m in ms]
        if all(b == a + 1 for a, b in zip(ls, ls[1:])):
            return ms
    m = opt_match(ln)
    return [m] if m else None


def _join_lines(lines):
    """Question text from raw lines: list items (A. / 1. / I. / List) start a new line, the rest is joined."""
    out = ""
    for l in lines:
        if not out:
            out = l
        elif re.match(r"^(?:[A-Ea-e][\.\)]|\d+[\.\)]|[IVX]{1,4}[\.\)]|List\b|Codes?\b)", l):
            out += "\n" + l
        else:
            out += " " + l
    return out


RE_META = re.compile(r"\[\s*(?:Question|Option)\s*(?:ID|Description)\b", re.I)  # "[Question ID = 12]", "[Option ID = 7]"


def clean_pages(pages, edge=4):
    """Join page texts, dropping running headers / footers ('Page 3 of 42'). Only the first and last few lines
    of a page can be a header / footer, and a line that repeats on many pages is only removed from there, so
    question text and "[Question ID = ..]" style tags in the middle of a page are never touched."""
    if len(pages) < 4:
        return "\n".join(pages)

    def norm(l):
        return re.sub(r"\d+", "#", re.sub(r"\s+", " ", l.strip().lower()))

    def candidate(l):
        return (len(l.strip()) >= 10 and not RE_META.search(l)
                and not (RE_OPT.match(l) or RE_Q.match(l) or RE_QNUM.match(l) or RE_ANS.match(l)))  # 10: 'Page 12 of 41'

    def edges(p):
        ls = [x for x in p.splitlines() if x.strip()]
        return ls[:edge] + ls[-edge:]

    cnt = Counter()
    for p in pages:
        for n in {norm(x) for x in edges(p) if candidate(x)}:
            cnt[n] += 1
    bad = {n for n, c in cnt.items() if c >= max(3, len(pages) // 2)}
    out = []
    for p in pages:
        ls = p.splitlines()
        nz = [i for i, x in enumerate(ls) if x.strip()]
        edge_idx = set(nz[:edge] + nz[-edge:])
        out += [x for i, x in enumerate(ls) if not (i in edge_idx and candidate(x) and norm(x) in bad)]
    return "\n".join(out)


def parse_answer(raw):
    """'B' -> ['B'] | 'A, C' -> ['A','C'] | '5.2' or '4.9 to 5.1' -> {'min','max'} | None"""
    raw = re.sub(r"^(?:option|opt\.?)\s*", "", raw.strip().rstrip(".").strip(), flags=re.I)
    if re.fullmatch(LETTERS, raw):
        return sorted(set(re.findall(r"[A-E]", raw.upper())))
    m = re.fullmatch(rf"({NUM})(?:\s*(?:to|[–—-])\s*({NUM}))?(?:\s+[^\d\s]\S*)?", raw)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2) or lo)
        return {"min": min(lo, hi), "max": max(lo, hi)}
    return None


def _finalize(q, a, kt=None):
    q.pop("_raw", None)
    q["question"] = q["question"].strip()
    if q["options"]:
        if isinstance(a, dict) and a["min"] == a["max"] and a["min"] == int(a["min"]) and 1 <= a["min"] <= len(q["options"]):
            a = [chr(64 + int(a["min"]))]  # key given as the option NUMBER
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
            n = int(m.group(1))
            key[n], types[n] = a, m.group(2).upper()
            idx.append(i)
    if len(key) < 3:
        return None
    start = idx[0]
    for j in range(max(0, start - 6), start):  # cut at the "Answer Key for ..." title just above the table
        if re.match(r"\s*answer\s*key\b", lines[j], re.I):
            start = j
            break
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
            take(rest)
            i += 1
            while i < len(lines) and PAIRS_ONLY.match(lines[i]):
                take(lines[i])
                i += 1
            continue
        keep.append(lines[i])
        i += 1
    if not key:
        j = len(keep)
        while j > 0 and PAIRS_ONLY.match(keep[j - 1]):
            j -= 1
        if len(keep) - j >= 3:
            for l in keep[j:]:
                take(l)
            keep = keep[:j]
    return key, keep, {}


RE_NQ = re.compile(r"^\s*(\d{1,3})\s*[\.\)]\s+(\S.*)$")  # "12. text" / "12) text" (only used by the numbered fallback)
RE_QID = re.compile(r"\[\s*Question\s*ID\s*=\s*\d+\s*\]\s*(?:\[\s*Question\s*Description\s*=[^\]]*\])?", re.I)
RE_OID = re.compile(r"\[\s*Option\s*ID\s*=\s*\d+\s*\]", re.I)
TQ, TO = "\u00a7Q\u00a7", "\u00a7O\u00a7"  # stand-ins for "[Question ID..]" and "[Option ID..]"


def _parse_lines(lines, numbered=False):
    """numbered=False: a question starts ONLY with Q / Q. / Q1 / Question (sub-parts "1." "2." never start one).
    numbered=True (fallback for papers numbered "1." / "1)"): a bare number starts a question only when it is the
    next number in order AND the previous question already has its options or answer, so statements
    '1. ... 2. ...' inside a question can never start a new one."""
    qs, cur, last = [], None, None

    def keep(ln):  # remember the raw line (used when 'A. B. C.' list items turn out not to be options)
        if cur is not None and cur.get("_raw") is not None:
            cur["_raw"].append(ln)

    for ln in lines:
        if RE_RANGE.match(ln):  # section heading like "Q.6 - Q.10 Carry TWO marks Each"
            last = None
            continue
        m = RE_QNUM.match(ln)
        if m:
            if cur and cur["question"].strip():
                qs.append(cur)
            cur = {"id": int(re.sub(r"\s+", "", m.group(1))), "type": "MCQ", "question": "",
                   "options": {}, "answer": None, "explanation": "", "_raw": []}
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
        ms = option_parts(ln)
        if cur is not None and ms and (cur["options"] or ms[0][0] == "A"):  # options start at A: "...also in B." is text
            if cur["options"] and ms[0][0] in cur["options"] and cur.get("_raw"):
                # "A. B. C. D." (or "(1) (2) (3)") were a LIST (match-the-following, statements); the real options
                # come now: put the list back into the question text
                cur["question"] = _join_lines(cur["_raw"])
                cur["options"], cur["_raw"] = {}, None
            else:
                keep(ln)
            for letter, text in ms:
                last = letter
                cur["options"][letter] = text
            continue
        m = RE_Q.match(ln)
        alone = None if m else RE_QALONE.match(ln)
        if m or alone:
            num = int(m.group(1) if m else alone.group(1))
            if alone or cur is None or cur["options"] or "_a" in cur or num == cur["id"] + 1:
                if cur and cur["question"].strip():
                    qs.append(cur)
                cur = {"id": num, "type": "MCQ", "question": m.group(2).strip() if m else "",
                       "options": {}, "answer": None, "explanation": "",
                       "_raw": [m.group(2).strip()] if m else []}
                last = "q"
                continue
        if numbered:
            m = RE_NQ.match(ln)
            if m:
                num = int(m.group(1))
                ready = cur is None or len(cur["options"]) >= 2 or "_a" in cur
                if ready and ((cur is None and num <= 3) or (cur is not None and num == cur["id"] + 1)):
                    if cur and cur["question"].strip():
                        qs.append(cur)
                    cur = {"id": num, "type": "MCQ", "question": m.group(2).strip(), "options": {},
                           "answer": None, "explanation": "", "_raw": [m.group(2).strip()]}
                    last = "q"
                    continue
        if cur is not None and last:  # wrapped line
            keep(ln)
            if last == "q":
                cur["question"] += " " + ln
            elif last == "exp":
                cur["explanation"] += " " + ln
            else:
                cur["options"][last] += " " + ln
    if cur and cur["question"].strip():
        qs.append(cur)
    return qs


RE_NUMOPT = re.compile(r"^\s*(\d)\s*[\.\)]\s*(.*)$")        # "1. text" / "2) text" / "3." (picture option)
RE_STEMNUM = re.compile(r"^\s*(\d{1,3})\s*[\.\)]\s*(.*)$")  # the question number in front of the first line
RE_STEMPAREN = re.compile(r"^\s*(\d{1,3})\s*\)\s*(.*)$")


def _mark(text):
    """Text of a paper that carries '[Question ID = 12][Question Description = ..]' and '[Option ID = 5]' tags
    (exam-portal exports): the tags become one-line markers. The tags are found in the whole text, because a tag
    can be split over two lines or two pages."""
    return RE_OID.sub(" " + TO + "\n", RE_QID.sub("\n" + TQ + "\n", text))


def _parse_marked(lines):
    """Layout:  <question text>  [Question ID]  1. option [Option ID]  2. option [Option ID] ...
    The question text is whatever stands before its [Question ID] tag, the options (numbered 1, 2, 3 ...) follow it
    and each is closed by its [Option ID] tag, so nothing needs to start with 'Q'. Options may be pictures (empty)."""
    n = len(lines)
    has_oid = any(TO in l for l in lines)

    def next_token(i):
        for j in range(i, n):
            if lines[j] == TQ:
                return TQ
            if TO in lines[j]:
                return TO
        return None

    qs, stem, i, expected = [], [], 0, 1
    while i < n:
        if lines[i] != TQ:
            stem.append(lines[i])
            i += 1
            continue
        i += 1
        opts, open_ = {}, None
        while i < n and lines[i] != TQ:
            l = lines[i]
            closed = TO in l
            body = l.replace(TO, "").strip()
            m = RE_NUMOPT.match(body)
            starts = bool(m) and int(m.group(1)) == len(opts) + 1 and len(opts) < 8
            if starts and has_oid:
                starts = next_token(i) == TO  # an option is closed by its [Option ID]; a question text is not
            if starts:
                open_ = chr(65 + len(opts))
                opts[open_] = m.group(2).strip()
                open_ = None if (closed or not has_oid) else open_
            elif open_ is not None:  # wrapped option text, up to its [Option ID]
                opts[open_] = (opts[open_] + " " + body).strip()
                if closed:
                    open_ = None
            elif closed and not body:  # a stray [Option ID]
                pass
            else:
                break  # first line of the next question
            i += 1
        # the question: skip page titles in front of its number, cut the number off
        idx = next((k for k, l in enumerate(stem) if (m := RE_STEMNUM.match(l)) and int(m.group(1)) == expected), None)
        if idx is None:
            idx = next((k for k, l in enumerate(stem) if RE_STEMPAREN.match(l)), None)
        body_lines = list(stem if idx is None else stem[idx:])
        num = expected
        if idx is not None:
            m = RE_STEMNUM.match(body_lines[0])
            num, body_lines[0] = int(m.group(1)), m.group(2)
        body_lines = [l for l in body_lines if l.strip()]
        text = _join_lines(body_lines)
        stem = []
        if not text and not opts:
            continue
        for L in opts:
            opts[L] = opts[L] or "(see figure)"
        qs.append({"id": num, "type": "MCQ", "question": text or "(see figure)", "options": opts,
                   "answer": None, "explanation": ""})
        expected = num + 1
    return qs


def _run(text, picture_opts, parser, use_key=True):
    """text -> (questions, info) with one line parser. info says whether an answer key was found, applied or rejected.
    picture_opts = {position in list: ['A','B','C','D']} for questions whose options are pictures: those options
    get the text "(see figure)" so the question, the answer key and the test all work normally."""
    raw = [l.strip() for l in text.splitlines() if l.strip()]
    key, lines, ktypes = _split_key(raw) if use_key else ({}, raw, {})
    qs = parser(lines)
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
                    or (isinstance(a, dict) and has and not (a["min"] == a["max"] and a["min"] == int(a["min"])
                                                              and 1 <= a["min"] <= len(q["options"])))):
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


def parse_with_info(text, picture_opts=None):
    """Returns (questions, info). Strategy 1 (always first): a question starts with Q / Q. / Q1 / Question.
    Only when that gives little or poor results the other layouts are tried (tagged exam-portal exports, then
    papers numbered '1.' / '1)'), and one of them is used only when it clearly finds more usable questions.
    info['dialect'] says which was used, info['numbered'] is True for the two fallbacks."""
    qs, info = _run(text, picture_opts, _parse_lines)
    info["dialect"], info["numbered"] = "q", False
    if len(qs) >= 5 and quality_ok(qs):
        return qs, info
    best, best_score = (qs, info), _score(qs)
    cands = []
    if len(RE_QID.findall(text)) >= 3:
        cands.append(("marked", lambda: _run(_mark(text), picture_opts, _parse_marked, use_key=False)))
    cands.append(("numbered", lambda: _run(text, picture_opts, lambda L: _parse_lines(L, True))))
    for name, fn in cands:
        q2, i2 = fn()
        sc = _score(q2)
        if sc > max(1.25 * best_score, 2):
            i2["dialect"], i2["numbered"] = name, True
            best, best_score = (q2, i2), sc
    return best


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
    seq = [q["id"] for q in qs]
    ids = set(seq)
    steps = [b - a for a, b in zip(seq, seq[1:])]
    sequential = bool(steps) and sum(1 for x in steps if x == 1) >= 0.6 * len(steps)  # a compilation (1, 3, 5, 10, 13...) is not sequential
    if sequential and len(qs) >= 5 and len(ids) < 0.9 * (max(ids) - min(ids) + 1):  # numbering has gaps: questions were lost
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
    """True only when the normal reader clearly failed: no questions, or fewer than 60% of them usable
    (90% for OCR text). A few flagged questions are NOT sent to the AI (it costs quota and often changes nothing):
    they are listed in the notes and marked in the Verify screen."""
    if not qs:
        return True
    check = structured if all(not q["answer"] for q in qs) else complete
    return sum(map(check, qs)) / len(qs) < (0.9 if strict else 0.6)


SUP = str.maketrans("0123456789+-\u2010\u2212\u2013n", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207a\u207b\u207b\u207b\u207b\u207f")
SUB = str.maketrans("0123456789+-\u2010\u2212\u2013", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u208a\u208b\u208b\u208b\u208b")


def _scripts(page, words):
    """{id(word): 'sup' | 'sub'} for small raised / lowered words that touch the word before them
    (the 2 in 'km2', the 4 in 'H2SO4', the -1 in 'LT-1'). Such digits are separate words in the PDF, which
    would otherwise read '1000 km 2 hilly' and 'H 2 SO 4'."""
    try:
        spans = [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
    except Exception:
        return {}
    sizes = Counter(round(s["size"]) for s in spans if len(s["text"].strip()) > 5)
    if not sizes:
        return {}
    main = sizes.most_common(1)[0][0]
    out = {}
    for s in spans:
        t = s["text"].strip()
        if not t or len(t) > 4 or s["size"] >= 0.8 * main:
            continue
        sx0, sy0, sx1, sy1 = s["bbox"]
        sc = (sy0 + sy1) / 2
        prev = [w for w in words if -1.5 <= sx0 - w[2] <= 0.8 and w[1] < sy1 and w[3] > sy0 and w[3] - w[1] > s["size"] * 1.05]
        me = [w for w in words if w[4] == t and sx0 - 0.8 <= (w[0] + w[2]) / 2 <= sx1 + 0.8 and sy0 - 0.8 <= (w[1] + w[3]) / 2 <= sy1 + 0.8]
        if not prev or not me:
            continue
        d = sc - (prev[-1][1] + prev[-1][3]) / 2
        if d < -0.8:
            out[id(me[0])] = "sup"
        elif d > 0.8:
            out[id(me[0])] = "sub"
    return out


def _join_words(ws, scripts):
    out, prev = "", None
    for w in ws:
        t, k = w[4], scripts.get(id(w))
        if k and out:
            ok = all(c in "0123456789+-\u2010\u2212\u2013" + ("n" if k == "sup" else "") for c in t)
            out += t.translate(SUP if k == "sup" else SUB) if ok else ("^" if k == "sup" else "_") + t
        elif (prev is not None and scripts.get(id(prev)) and w[0] - prev[2] < 1.5
              and (t[:1] in ",.;:)]" or t[:1].isupper() or t[:1].isdigit())):
            out += t  # 'SO' right after the subscript of 'H2', but 'H2SO4 and' keeps its space
        else:
            out += (" " if out else "") + t
        prev = w
    return out


def rows_text(page, tol=3):
    """Rebuild text row by row from word positions. Fixes tables where every cell ('(A)' | 'Accelerate')
    is otherwise returned on its own line."""
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    scripts = _scripts(page, words)
    rows = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if rows and abs(rows[-1][0] - yc) <= tol:
            rows[-1][1].append(w)
        else:
            rows.append([yc, [w]])
    # [centre y, text, top, bottom, left, right] per row
    lines = [[yc, _join_words(sorted(ws, key=lambda w: w[0]), scripts),
              min(w[1] for w in ws), max(w[3] for w in ws), min(w[0] for w in ws), max(w[2] for w in ws)]
             for yc, ws in rows]
    out, pos = [], {}  # pos: row index -> its place in out
    for k in range(len(lines)):
        yc, t, y0, y1, x0, x1 = lines[k]
        if re.fullmatch(r"Q\.?\s*\d+", t.strip()):  # a lone "Q.46" label sitting a few points off its text line
            # text to the right of the label that overlaps it vertically: the label belongs to the TOPMOST such row
            # (in question-sheet PDFs the label can sit lower than the first line of its cell)
            cand = [j for j in range(len(lines)) if j != k and lines[j][4] > x1 - 1
                    and lines[j][3] > y0 + 0.5 and lines[j][2] < y1 - 0.5]
            if cand and min(cand) < k and min(cand) in pos:
                out[pos[min(cand)]][1] = t.strip() + " " + out[pos[min(cand)]][1]
                continue
            if cand and min(cand) > k:
                lines[min(cand)][1] = t.strip() + " " + lines[min(cand)][1]
                continue
            dprev = yc - out[-1][0] if out else 99
            dnext = lines[k + 1][0] - yc if k + 1 < len(lines) else 99
            if min(dprev, dnext) <= 12:
                if dprev <= dnext:
                    out[-1][1] = t.strip() + " " + out[-1][1]
                else:
                    lines[k + 1][1] = t.strip() + " " + lines[k + 1][1]
                continue
        pos[k] = len(out)
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
    return len(re.sub(r"\s+", " ", t)) < 300 and bool(page.get_image_info())  # a picture with only a few header words of real text


def _score(qs):
    return sum(2 if complete(q) else (1 if len(q["options"]) >= 2 else 0) for q in qs)


def best_text_parse(doc, pages, ocr_idx=()):
    """Read the PDF two ways - normal text order, and row by row (fixes table layouts) - and keep
    whichever version gives more usable questions."""
    text = clean_pages(pages)
    qs, info = parse_with_info(text) if text.strip() else ([], {})
    if len(qs) >= 5 and quality_ok(qs) and os.getenv("ALWAYS_ROW_PARSE") != "1":
        return text, qs, info  # already good: skip the second (row by row) read, it cannot do better
    rows = [pages[i] if i in ocr_idx else rows_text(doc[i]) for i in range(len(pages))]
    t2 = clean_pages(rows)
    q2, i2 = parse_with_info(t2) if t2.strip() else ([], {})
    return (t2, q2, i2) if _score(q2) > _score(qs) else (text, qs, info)