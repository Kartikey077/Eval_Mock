"""OCR for scanned pages (Tesseract). Needs the Tesseract program installed."""
import io, os, shutil


def _setup():
    import pytesseract
    cmd = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
    if not cmd:
        for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
            if os.path.exists(p):
                cmd = p
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return pytesseract


def available():
    try:
        _setup().get_tesseract_version()
        return True
    except Exception:
        return False


def page_text(page):
    import pymupdf
    from PIL import Image
    pt = _setup()
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY, alpha=False)  # grey = 1/3 of the memory of colour
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    pix = None  # free the pixmap now
    try:
        return pt.image_to_string(img, config="--psm 4")
    finally:
        img.close()
        del img