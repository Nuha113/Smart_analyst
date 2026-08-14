"""Stable OCR import path."""

from pathlib import Path

from backend.app.services.ocr_service import OCRService


def read(file_path):
    path = Path(file_path)
    mime = "application/pdf" if path.suffix.lower() == ".pdf" else "image/jpeg"
    return OCRService().extract(path, mime, path.stat().st_size)["texts"]
