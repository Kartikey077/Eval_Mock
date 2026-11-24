"""The real work (PDF reading, OCR, pictures, AI). It runs ONLY inside a short-lived child process (see runner.py),
so every byte of memory the PDF needed goes back to the system when the child exits. The web server itself never
imports pymupdf / PIL / Tesseract, which is why it stays small while idle."""
import difflib
import logging
import os
import re

import pymupdf

import llm
import ocr
import snapshot
from figures import Pages, find_figures, question_bands
from layout import find_layout
from memutil import release_memory
from quizparse import (best_text_parse, complete, has_options, image_based, needs_ai,
                       parse_with_info, problems, quality_ok)

log = logging.getLogger("mocktest")
MAX_PAGES = int(os.getenv("MAX_PAGES", "70"))
AI_USED = [False]  # set when this job called the AI (the web server counts it against the daily limit)


def jsonify(**kw):  # same call style as Flask's jsonify; the runner turns the dict into the HTTP response
    return kw


def _norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()[:80]


def _similar(a, b):
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def carry_images(old, new):
    """After the AI replaced every question, give the pictures and layout (tables, code) back to the
    matching questions (by text)."""
    pics = [q for q in old if q.get("images") or q.get("blocks")]
    for n in new:
        best, score = None, 0.0
        for q in pics:
            s = _similar(q["question"], n["question"])
            if s > score:
                best, score = q, s
        if best is not None and score >= 0.6:
            for key in ("images", "blocks"):
                if best.get(key):
                    n[key] = best[key]


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
            for key in ("images", "blocks"):
                if q.get(key):
                    n[key] = q[key]
            qs[i] = n
            taken.add(best)
            replaced.append(i + 1)
    return replaced


def pictures_response(doc, pages, filename, why=""):
    """Picture mode: every question is cut out as one picture (no text reading, no OCR, nothing can be lost)."""
    qs, notes = snapshot.snapshot_questions(doc, pages)
    if not qs:
        return jsonify(error="This PDF looks blank: nothing could be cut out of its pages."), 422
    if why:
        notes.insert(0, why)
    return jsonify(filename=filename, questions=qs, method="Pictures", notes=notes)




def extract(path, filename, force=False, mode='', ai_ok=True):
    notes, used = [], set()
    try:
        doc = pymupdf.open(path)  # opened from disk: the PDF is never held in memory as one big bytes object
    except Exception:
        log.exception("Could not open PDF %s", filename)
        return jsonify(error="This file could not be read as a PDF."), 400
    if doc.page_count == 0:
        return jsonify(error="This PDF has no pages."), 400
    if doc.page_count > MAX_PAGES:  # checked BEFORE any text is read
        return jsonify(error=f"This PDF has {doc.page_count} pages. The limit is {MAX_PAGES}."), 400
    try:
        pages = [p.get_text("text") for p in doc]
    except Exception:
        log.exception("Could not read PDF text %s", filename)
        return jsonify(error="This file could not be read as a PDF."), 400
    log.info("Extract: %s, %d pages, force_ai=%s", filename, len(pages), force)
    if mode == "pictures":  # the user chose picture mode (best for maths papers)
        return pictures_response(doc, pages, filename)

    # 1. Scanned pages -> OCR
    blank = [i for i, t in enumerate(pages) if image_based(doc[i], t)]
    if blank:
        if ocr.available():
            try:
                for n, i in enumerate(blank):
                    pages[i] = ocr.page_text(doc[i])
                    if n % 8 == 7:
                        release_memory()  # long scans: do not let buffers pile up
                used.add("OCR")
            except Exception as e:
                log.exception("OCR failed")
                notes.append("OCR failed: " + str(e)[:120])
        else:
            notes.append(f"{len(blank)} scanned page(s) found, but Tesseract OCR is not installed.")

    # 2. Rule-based parsing
    text, qs, info = best_text_parse(doc, pages, set(blank) if "OCR" in used else set())

    # 2b. Diagrams and layout: cut pictures out of the pages (questions whose options are pictures get
    #     "(see figure)" options) and rebuild tables / program code / lists so they are readable
    try:
        pg = Pages(doc)  # one page cache shared by the three steps
        nb = bool(info.get("numbered"))  # paper numbered 1) 2) 3) instead of Q1.
        sb = question_bands(doc, qs, set(blank), pg, nb)
        images, pic = find_figures(doc, qs, set(blank), starts_bands=sb, pg=pg, numbered=nb)
        if pic:
            qs, info = parse_with_info(text, pic)
        layout = find_layout(doc, qs, set(blank), sb, images, pg=pg, numbered=nb)
        pg.clear()
        del pg
        for i, q in enumerate(qs):
            if i in images:
                q["images"] = images[i]
            if i in layout:
                q["blocks"] = layout[i]
        log.info("Figures: %d with pictures (%d with picture options); layout: %d with tables/code/lists",
                 len(images), len(pic), len(layout))
        if pic:
            notes.append("Question(s) " + ", ".join(map(str, sorted(i + 1 for i in pic)[:12]))
                         + " use pictures as options. The picture is shown with the question and the options are labelled A, B, C, D.")
    except Exception:
        log.exception("Figure / layout detection failed")

    # 3. AI fallback, when the result is missing, incomplete, or has fewer options than the rest of the paper
    strict = "OCR" in used
    rules_n = len(qs)
    inc_ids, few_ids = problems(qs)
    unreadable = len(blank) * 2 >= len(pages) and "OCR" not in used
    want_ai = force or unreadable or needs_ai(qs, strict)
    log.info("Rules found %d questions; flagged %d incomplete, %d few-option; want_ai=%s",
             rules_n, len(inc_ids), len(few_ids), want_ai)
    if want_ai:
        if llm.available() and not ai_ok:
            notes.append("Daily AI limit reached for your connection. Try again tomorrow.")
        elif llm.available():
            AI_USED[0] = True
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

    # --- guard: nothing parsed -> picture mode instead of an error (this is the "never fails" step) ---
    if not qs:
        try:
            return pictures_response(doc, pages, filename, "The text could not be read as questions, so each question was cut out as a picture.")
        except Exception:
            log.exception("Picture mode failed")
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
    if not quality_ok(qs, "OCR" in used):
        notes.append('Many questions look incomplete. For maths or symbol-heavy papers press "Cut as pictures": '
                     'it keeps every question and option exactly as printed.')
    method = ("OCR" if "OCR" in used else "Pictures" if unreadable else "Text") + (" + AI" if "AI" in used else "")
    return jsonify(filename=filename, questions=qs, method=method, notes=notes)



def suggest(items, ai_ok=True):
    if not llm.available():
        return jsonify(error="AI is not set up (no API key)."), 400
    if not ai_ok:
        return jsonify(error="Daily AI limit reached. Try again tomorrow."), 429
    if not isinstance(items, list):
        return jsonify(error="Send a JSON body like {\"questions\": [...]}."), 400
    AI_USED[0] = True
    try:
        return jsonify(answers=llm.suggest(items[:100]))
    except Exception as e:
        log.exception("Suggest failed")
        return jsonify(error=str(e)[:200]), 502