"""Give memory back after heavy work (PDF pages, pixmaps, OCR buffers)."""
import ctypes
import gc
import sys

import pymupdf


def release_memory():
    gc.collect()
    try:
        pymupdf.TOOLS.store_shrink(100)  # empty MuPDF's internal cache
    except Exception:
        pass
    if sys.platform.startswith("linux"):  # glibc keeps freed heap; ask it to hand it back to the OS
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass