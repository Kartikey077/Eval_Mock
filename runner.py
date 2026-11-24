"""Runs one job (a PDF extraction or an AI suggestion) in a SEPARATE short-lived process.

Why: Python and MuPDF never give memory back to the system, so a web process that reads PDFs stays big for ever
(and fragments over days). A child process that ends after every job returns ALL its memory, so the web server
stays at its small idle size. A watchdog also kills a job that grows past MAX_JOB_MB, so one huge PDF can never
take the whole 1 GB machine down. This file imports nothing heavy: the heavy imports happen in the child only.
"""
import logging
import multiprocessing as mp
import os
import signal
import tempfile
import threading
import time

log = logging.getLogger("mocktest")
MAX_JOB_MB = int(os.getenv("MAX_JOB_MB", "750"))    # kill a job that uses more than this (all its processes together)
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "900"))  # seconds (keep below the gunicorn --timeout)
QUEUE_WAIT = int(os.getenv("QUEUE_WAIT", "900"))     # how long a second user waits for the one job slot
_slot = threading.Semaphore(int(os.getenv("MAX_JOBS", "1")))  # 1 = one PDF at a time: the safe choice for 1 GB


def _child(conn, kind, kwargs):
    """Child process entry. Imports the heavy modules here, runs the job, sends (payload, status, ai_used) back."""
    try:
        os.setsid()  # own process group: the watchdog can kill the job and Tesseract together
    except Exception:
        pass
    os.environ.setdefault("PAGE_CACHE", "100")
    os.environ.setdefault("OMP_THREAD_LIMIT", "12")  # drawings are the heavy part: keep fewer
    try:
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        import jobs
        res = getattr(jobs, kind)(**kwargs)
        payload, code = res if isinstance(res, tuple) else (res, 200)
        conn.send((payload, code, jobs.AI_USED[0]))
    except MemoryError:
        conn.send(({"error": "This PDF needs more memory than the server has. Try a PDF with fewer pages."}, 413, False))
    except Exception:
        logging.getLogger("mocktest").exception("Job %s failed", kind)
        conn.send(({"error": "Something went wrong on the server. Please try again."}, 500, False))
    finally:
        conn.close()


def _rss_mb(pid):
    """Memory (MB) of a process plus its children (Tesseract). Linux only; returns 0 elsewhere."""
    total, want = 0, {pid}
    try:
        procs = {}
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    st = open(f"/proc/{d}/status").read()
                except OSError:
                    continue
                ppid = int(st.split("PPid:")[1].split()[0])
                rss = int(st.split("VmRSS:")[1].split()[0]) if "VmRSS:" in st else 0
                procs[int(d)] = (ppid, rss)
        grew = True
        while grew:
            grew = False
            for p, (pp, _r) in procs.items():
                if pp in want and p not in want:
                    want.add(p)
                    grew = True
        total = sum(procs[p][1] for p in want if p in procs)
    except Exception:
        return 0
    return total // 1024


def run_job(kind, **kwargs):
    """Run jobs.<kind>(**kwargs) in a child process. Returns (payload_dict, http_status, ai_used)."""
    if not _slot.acquire(timeout=QUEUE_WAIT):
        return {"error": "The server is busy with another PDF. Please try again in a minute."}, 503, False
    proc = None
    try:
        ctx = mp.get_context("spawn")  # a fresh, clean interpreter: nothing inherited from the web server
        recv, send = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_child, args=(send, kind, kwargs), daemon=True)
        proc.start()
        send.close()
        t0, result = time.time(), None
        while True:
            if recv.poll(0.4):
                try:
                    result = recv.recv()
                except EOFError:
                    pass
                break
            if not proc.is_alive():
                break
            if time.time() - t0 > JOB_TIMEOUT:
                log.warning ("Job %s timed out", kind)
                _kill(proc)
                return {"error": "This PDF took too long to process. Try a smaller PDF."}, 504, False
            mb = _rss_mb(proc.pid)
            if mb > MAX_JOB_MB:
                log.warning("Job %s used %d MB (limit %d): stopped", kind, mb, MAX_JOB_MB)
                _kill(proc)
                return {"error": "This PDF is too big for this server's memory. Try a PDF with fewer pages."}, 413, False
        proc.join(5)
        if result is None:
            return {"error": "The PDF could not be processed (the worker stopped). Try a smaller PDF."}, 500, False
        return result
    finally:
        if proc is not None:
            _kill(proc)
        _slot.release()


def _kill(proc):
    try:
        if proc.is_alive():
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # the job and anything it started (Tesseract)
            except Exception:
                proc.kill()
        proc.join(2)
    except Exception:
        pass


def save_upload(file_storage):
    """Stream the upload to a temp file (never as one big bytes object in the web process). Returns the path."""
    fd, path = tempfile.mkstemp(suffix=".pdf", prefix="mock_")
    os.close(fd)
    file_storage.save(path)
    return path