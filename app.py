"""Mock Test Maker - Version 3.3  (small web server; every PDF is processed in a short-lived child process)
Local run:   flask --app app run --debug      ->  http://localhost:5000   (debug ONLY on your own computer)
Deployed:    gunicorn -c gunicorn.conf.py app:app
The web server imports nothing heavy (no PDF / image / OCR libraries): it uses only a few tens of MB while idle.
"""
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

import llm
import runner

BASE = Path(__file__).resolve().parent
envfile = BASE / ".env"  # optional: GEMINI_API_KEY=...  /  TESSERACT_CMD=...
if envfile.exists():
    for line in envfile.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mocktest")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

VERSION = "3.3"
AI_LIMIT = int(os.getenv("AI_DAILY_LIMIT", "40"))  # AI calls per visitor per day (protects the free quota)
_uses = {}


def ai_left(ip):
    day = time.strftime("%Y-%m-%d")
    for k in [k for k, (d, _) in _uses.items() if d != day]:  # drop old days so the dict cannot grow forever
        del _uses[k]
    return _uses.get(ip, (day, 0))[1] < AI_LIMIT


def ai_count(ip):
    day = time.strftime("%Y-%m-%d")
    _, n = _uses.get(ip, (day, 0))
    _uses[ip] = (day, n + 1)


def ocr_ready():
    """Cheap check (no heavy import): is the Tesseract program installed?"""
    cmd = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
    if cmd and (shutil.which(cmd) or os.path.exists(cmd)):
        return True
    return any(os.path.exists(p) for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                                           r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"))


@app.errorhandler(Exception)
def on_error(e):
    if isinstance(e, HTTPException):  # 404, 413 etc. keep their normal response
        return e
    log.exception("Unhandled error on %s", request.path)
    return jsonify(error="Something went wrong on the server. Please try again."), 500


@app.get("/")
def home():
    for p in (BASE / "static" / "index.html", BASE / "index.html"):
        if p.exists():
            return send_file(p)
    return "index.html not found. Put it in: " + str(BASE / "static" / "index.html"), 404


@app.get("/api/status")
def status():
    return jsonify(ocr=ocr_ready(), ai=llm.available(), version=VERSION)


JOBS = {}  # job id -> {"state", "t", "payload", "code"}


def _work(jid, path, filename, mode, ip):
    try:
        payload, code, ai_used = runner.run_job(
            "extract", path=path, filename=filename, mode=mode, ai_ok=ai_left(ip))
        if ai_used:
            ai_count(ip)
        res = {"payload": payload, "code": code}
    except Exception:
        log.exception("Background extract failed")
        res = {"payload": {"error": "Something went wrong on the server. Please try again."}, "code": 500}
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    JOBS[jid].update(res, state="done", t=time.time())


@app.post("/api/extract")
def extract():
    f = request.files.get("pdf")
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify(error="Choose a PDF file to upload."), 400
    for k in [k for k, j in JOBS.items() if time.time() - j["t"] > 1800]:  # forget old jobs
        JOBS.pop(k, None)
    path = runner.save_upload(f)
    jid = uuid.uuid4().hex
    JOBS[jid] = {"state": "running", "t": time.time()}
    threading.Thread(target=_work, daemon=True, args=(
        jid, path, f.filename, request.form.get("mode", ""), request.remote_addr)).start()
    return jsonify(job=jid), 202


@app.get("/api/job/<jid>")
def job(jid):
    j = JOBS.get(jid)
    if not j:
        return jsonify(error="Job not found (the server restarted). Please upload again."), 404
    if j["state"] != "done":
        return jsonify(state="running")
    return jsonify(state="done", code=j["code"], payload=j["payload"])


@app.post("/api/suggest")
def suggest():
    if not llm.available():
        return jsonify(error="AI is not set up (no API key)."), 400
    data = request.get_json(silent=True)
    items = data.get("questions") if isinstance(data, dict) else None
    payload, code, ai_used = runner.run_job("suggest", items=items, ai_ok=ai_left(request.remote_addr))
    if ai_used:
        ai_count(request.remote_addr)
    return jsonify(payload), code


if __name__ == "__main__":
    app.run()  # no debug=True. For local debugging use:  flask --app app run --debug