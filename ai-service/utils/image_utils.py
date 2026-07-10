"""
Image validation and I/O utilities.

Handles file-type checks and conversion of uploaded bytes into a format
compatible with Keras `load_img`, without loading the entire file twice.
"""

import io
import logging
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException, UploadFile

from app.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_IMAGE_MIME_TYPES,
    MAX_UPLOAD_SIZE_BYTES,
)

logger = logging.getLogger(__name__)


def _normalize_extension(filename: str | None) -> str:
    """Extract and lowercase the file extension from an upload filename."""
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def validate_image_upload(file: UploadFile) -> None:
    """
    Validate that the uploaded file is an allowed image type.

    Raises HTTP 400 when the filename extension or MIME type is not permitted.
    """
    extension = _normalize_extension(file.filename)
    content_type = (file.content_type or "").lower()

    extension_valid = extension in ALLOWED_IMAGE_EXTENSIONS
    mime_valid = content_type in ALLOWED_IMAGE_MIME_TYPES

    if not extension_valid and not mime_valid:
        logger.warning(
            "Rejected upload: filename=%s content_type=%s",
            file.filename,
            file.content_type,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid file type. Only JPG, JPEG, and PNG images are accepted. "
                f"Received extension='{extension}' content_type='{content_type}'."
            ),
        )


async def read_upload_stream(file: UploadFile) -> BinaryIO:
    """
    Read an UploadFile in chunks and return a seekable in-memory buffer.

    Uses chunked reads to avoid loading oversized files entirely into memory
    at once, while still enforcing the configured maximum upload size.
    """
    buffer = io.BytesIO()
    total_bytes = 0
    chunk_size = 64 * 1024  # 64 KB chunks

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break

        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"File too large. Maximum allowed size is "
                    f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
                ),
            )
        buffer.write(chunk)

    if total_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    buffer.seek(0)
    return buffer
