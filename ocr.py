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
    from PIL import Image
    pt = _setup()
    img = Image.open(io.BytesIO(page.get_pixmap(dpi=200).tobytes("png")))
    return pt.image_to_string(img, config="--psm 4")
