"""FastAPI web server for the kinship-verification model."""

import io
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from predict import predict


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="Kinship Verification")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def validate_image(contents: bytes, upload: UploadFile) -> None:
    if not contents:
        raise HTTPException(status_code=400, detail=f"{upload.filename or 'Image'} is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Each image must be 10 MB or smaller.")
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Use a JPG, PNG, or WebP image.")
    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="One of the uploads is not a valid image.") from error


def save_temp_image(contents: bytes, filename: str | None) -> str:
    suffix = Path(filename or "image.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        temporary_file.write(contents)
        return temporary_file.name


@app.post("/api/predict")
async def make_prediction(
    first_image: UploadFile = File(...),
    second_image: UploadFile = File(...),
) -> dict[str, str | float]:
    first_contents, second_contents = await first_image.read(), await second_image.read()
    validate_image(first_contents, first_image)
    validate_image(second_contents, second_image)

    first_path = save_temp_image(first_contents, first_image.filename)
    second_path = save_temp_image(second_contents, second_image.filename)
    try:
        return predict(first_path, second_path)
    except Exception as error:
        raise HTTPException(status_code=500, detail="Prediction could not be completed. Please try different images.") from error
    finally:
        for path in (first_path, second_path):
            if os.path.exists(path):
                os.remove(path)
