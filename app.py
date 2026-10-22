"""Mock Test Maker - Version 2.8  (text/OCR -> rules -> AI fallback; MCQ, MSQ, NAT)
Local run:   flask --app app run --debug      ->  http://localhost:5000   (debug ONLY on your own computer)
Deployed:    gunicorn app:app --workers 1 --threads 4 --timeout 180 --access-logfile - --error-logfile -
"""
import difflib
import logging
import os
import re
import time
from pathlib import Path

import pymupdf
from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

import llm
import ocr
from figures import find_figures
from quizparse import (best_text_parse, complete, has_options, image_based, needs_ai,
                       parse_with_info, problems, quality_ok)

BASE = Path(__file__).resolve().parent
envfile = BASE / ".env"  # optional: ANTHROPIC_API_KEY=sk-ant-...  /  TESSERACT_CMD=...
if envfile.exists():
    for line in envfile.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

# Logging: set LOG_LEVEL=DEBUG in the host's environment variables when you need more detail.
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mocktest")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

VERSION = "2.8"
AI_LIMIT = int(os.getenv("AI_DAILY_LIMIT", "40"))  # AI calls per visitor per day (protects the free quota)
MAX_PAGES = int(os.getenv("MAX_PAGES", "150"))
_uses = {}


def ai_allowed(ip):
    day = time.strftime("%Y-%m-%d")
    for k in [k for k, (d, _) in _uses.items() if d != day]:  # drop old days so the dict cannot grow forever
        del _uses[k]
    d, n = _uses.get(ip, (day, 0))
    if n >= AI_LIMIT:
        return False
    _uses[ip] = (day, n + 1)
    return True


def _norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()[:80]


def _similar(a, b):
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def carry_images(old, new):
    """After the AI replaced every question, give the pictures back to the matching questions (by text)."""
    pics = [q for q in old if q.get("images")]
    for n in new:
        best, score = None, 0.0
        for q in pics:
            s = _similar(q["question"], n["question"])
            if s > score:
                best, score = q, s
        if best is not None and score >= 0.6:
            n["images"] = best["images"]


def merge_ai(qs, new, target):
    """Replace flagged questions with the AI's version, matching by QUESTION TEXT, never by number.
    (The AI numbers questions 1,2,3... by position; the paper may start elsewhere or have gaps.)
    The replaced question keeps the paper's own number. Returns the list of replaced numbers."""
    taken, replaced = set(), []
    for i, q in enumerate(qs):
        if i + 1 not in target:  # target = 1-based positions from problems()
            continue
        best, score = None, 0.0
        for j, n in enumerate(new):
            if j in taken:
                continue
            s = _similar(q["question"], n["question"])
            if s > score:
                best, score = j, s
        if best is not None and score >= 0.5 and complete(new[best]):
            n = new[best]
            n["id"] = q["id"]
            if q.get("images"):
                n["images"] = q["images"]
            qs[i] = n
            taken.add(best)
            replaced.append(i + 1)
    return replaced


@app.errorhandler(Exception)
def on_error(e):
    if isinstance(e, HTTPException):  # 404, 413 etc. keep their normal response
        return e
    log.exception("Unhandled error on %s", request.path)  # full traceback goes to the server log
    return jsonify(error="Something went wrong on the server. Please try again."), 500


@app.get("/")
def home():
    for p in (BASE / "static" / "index.html", BASE / "index.html"):
        if p.exists():
            return send_file(p)
    return "index.html not found. Put it in: " + str(BASE / "static" / "index.html"), 404


@app.get("/api/status")
def status():
    return jsonify(ocr=ocr.available(), ai=llm.available(), provider=llm.provider(), version=VERSION)


@app.post("/api/extract")
def extract():
    f = request.files.get("pdf")
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify(error="Choose a PDF file to upload."), 400
    notes, used = [], set()
    force = request.form.get("force_ai") == "1"
    try:
        doc = pymupdf.open(stream=f.read(), filetype="pdf")
        pages = [p.get_text("text") for p in doc]
    except Exception:
        log.exception("Could not open PDF %s", f.filename)
        return jsonify(error="This file could not be read as a PDF."), 400
    if not pages:
        return jsonify(error="This PDF has no pages."), 400
    if len(pages) > MAX_PAGES:
        return jsonify(error=f"This PDF has {len(pages)} pages. The limit is {MAX_PAGES}."), 400
    log.info("Extract: %s, %d pages, force_ai=%s", f.filename, len(pages), force)

    # 1. Scanned pages -> OCR
    blank = [i for i, t in enumerate(pages) if image_based(doc[i], t)]
    if blank:
        if ocr.available():
            try:
                for i in blank:
                    pages[i] = ocr.page_text(doc[i])
                used.add("OCR")
            except Exception as e:
                log.exception("OCR failed")
                notes.append("OCR failed: " + str(e)[:120])
        else:
            notes.append(f"{len(blank)} scanned page(s) found, but Tesseract OCR is not installed.")

    # 2. Rule-based parsing
    text, qs, info = best_text_parse(doc, pages, set(blank) if "OCR" in used else set())

    # 2b. Diagrams: cut pictures out of the pages; questions whose options are pictures get "(see figure)" options
    try:
        images, pic = find_figures(doc, qs, set(blank))
        if pic:
            qs, info = parse_with_info(text, pic)
        for i, q in enumerate(qs):
            if i in images:
                q["images"] = images[i]
        log.info("Figures: %d question(s) with pictures, %d with picture options", len(images), len(pic))
        if pic:
            notes.append("Question(s) " + ", ".join(map(str, sorted(i + 1 for i in pic)[:12]))
                         + " use pictures as options. The picture is shown with the question and the options are labelled A, B, C, D.")
    except Exception:
        log.exception("Figure detection failed")

    # 3. AI fallback, when the result is missing, incomplete, or has fewer options than the rest of the paper
    strict = "OCR" in used
    rules_n = len(qs)
    inc_ids, few_ids = problems(qs)
    unreadable = len(blank) * 2 >= len(pages) and "OCR" not in used
    want_ai = force or unreadable or needs_ai(qs, strict)
    log.info("Rules found %d questions; flagged %d incomplete, %d few-option; want_ai=%s",
             rules_n, len(inc_ids), len(few_ids), want_ai)
    if want_ai:
        if llm.available() and not ai_allowed(request.remote_addr):
            notes.append("Daily AI limit reached for your connection. Try again tomorrow.")
        elif llm.available():
            try:
                new = llm.parse_text(text, expect=rules_n) if (has_options(text) and not (force and blank)) else []
                if not new or not quality_ok(new, strict) or (force and blank):
                    pics = llm.parse_images(doc, blank)
                    if pics and (force or len(pics) > len(new)):
                        new = pics
                if new:
                    if force:
                        carry_images(qs, new)
                        qs, used = new, used | {"AI"}
                        notes.append("AI re-read the PDF at your request.")
                    else:
                        target = set(inc_ids) | set(few_ids)
                        replaced = merge_ai(qs, new, target)
                        if replaced:
                            used = used | {"AI"}
                            notes.append(f"AI re-read {len(replaced)} flagged question(s).")
                        elif target:
                            notes.append("AI could not be matched safely to the flagged questions, so they were left unchanged.")
                        if len(new) > len(qs):
                            notes.append(f'The AI found {len(new) - len(qs)} more question(s) than the normal reader. '
                                         'Press "Re-read PDF with AI" to use the AI version.')
            except Exception as e:
                log.exception("AI parsing failed")
                notes.append("AI parsing failed: " + str(e)[:150])
        else:
            notes.append("Parsing looks incomplete. Add a free GEMINI_API_KEY to a .env file to enable AI parsing for hard PDFs.")

    # --- guard: nothing parsed -> clear error ---
    if not qs:
        if len(blank) * 2 >= len(pages):
            todo = ("Install Tesseract OCR, or add a free GEMINI_API_KEY to the .env file so the AI can read the pictures."
                    if not (ocr.available() or llm.available()) else "OCR/AI could not read the questions from the pictures. Try a clearer or larger scan.")
            notes.insert(0, "This PDF is made of page pictures, not text. " + todo)
        return jsonify(error=" ".join(notes) or "No questions found. Check that the PDF numbers its questions and lists options as (A) (B) (C) (D)."), 422

    # --- notes on the (possibly merged) result ---
    bad = sum(1 for q in qs if not complete(q))
    inc_ids, few_ids = problems(qs)
    if info.get("mismatch"):
        notes.append("An answer key was found at the end of the PDF, but it does not match this paper (for example questions "
                     + ", ".join(map(str, info["mismatch"][:5])) + " have a different type than the key says). "
                     "It may belong to a different exam or year, so it was NOT applied.")
    elif info.get("key_applied"):
        notes.append("Answers were read from the answer key in the PDF.")
    if all(not q["answer"] for q in qs):
        if not info.get("mismatch"):
            notes.append("No answers were found in this PDF. Set them below" + (' or use "Suggest missing answers with AI".' if llm.available() else "."))
    elif bad:
        notes.append(f"{bad} question(s) look incomplete (missing answer or options). They are flagged below.")
    if few_ids:
        notes.append("Questions " + ", ".join(map(str, few_ids[:12]))
                     + " have fewer options than the rest of the paper; check them manually"
                     + (' or press "Re-read PDF with AI".' if llm.available() else "."))
    if "OCR" in used and "AI" not in used:
        notes.append("Read with OCR: symbols and maths may be misread, so check the questions" + (' or press "Re-read PDF with AI".' if llm.available() else "."))
    method = ("OCR" if "OCR" in used else "Pictures" if unreadable else "Text") + (" + AI" if "AI" in used else "")
    doc.close()
    return jsonify(filename=f.filename, questions=qs, method=method, notes=notes)


@app.post("/api/suggest")
def suggest():
    if not llm.available():
        return jsonify(error="AI is not set up (no API key)."), 400
    if not ai_allowed(request.remote_addr):
        return jsonify(error="Daily AI limit reached. Try again tomorrow."), 429
    data = request.get_json(silent=True)
    items = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return jsonify(error="Send a JSON body like {\"questions\": [...]}."), 400
    try:
        return jsonify(answers=llm.suggest(items[:100]))
    except Exception as e:
        log.exception("Suggest failed")
        return jsonify(error=str(e)[:200]), 502


if __name__ == "__main__":
    app.run()  # no debug=True. For local debugging use:  flask --app app run --debug