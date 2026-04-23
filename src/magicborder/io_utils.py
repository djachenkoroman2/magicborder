from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from PyQt5.QtGui import QImage, QPixmap

from .models import Annotation

SUPPORTED_RASTER_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(slots=True)
class LoadedImage:
    path: Path
    qimage: QImage
    pixmap: QPixmap
    rgb_array: np.ndarray
    width: int
    height: int


def image_open_filter() -> str:
    patterns = " ".join(f"*{suffix}" for suffix in SUPPORTED_RASTER_SUFFIXES)
    return f"Raster images ({patterns});;All files (*)"


def image_save_filter() -> str:
    return (
        "PNG image (*.png);;"
        "JPEG image (*.jpg *.jpeg);;"
        "Bitmap image (*.bmp);;"
        "TIFF image (*.tif *.tiff)"
    )


def load_raster_image(path: str | Path) -> LoadedImage:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Файл не найден: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_RASTER_SUFFIXES:
        raise ValueError("Неподдерживаемый формат изображения.")

    try:
        with Image.open(image_path) as pil_image:
            rgba_image = pil_image.convert("RGBA")
            rgb_image = pil_image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Не удалось распознать файл как растровое изображение.") from exc
    except OSError as exc:
        raise ValueError(f"Не удалось открыть изображение: {exc}") from exc

    rgb_array = np.array(rgb_image, dtype=np.uint8)
    return loaded_image_from_rgb_array(image_path, rgb_array, rgba_image=rgba_image)


def save_annotation(path: str | Path, annotation: Annotation) -> None:
    annotation_path = Path(path)
    payload = json.dumps(annotation.to_dict(), ensure_ascii=False, indent=2)
    annotation_path.write_text(payload, encoding="utf-8")


def load_annotation(path: str | Path) -> Annotation:
    annotation_path = Path(path)
    try:
        raw_text = annotation_path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Файл аннотации не найден: {annotation_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON: {exc}") from exc
    return Annotation.from_dict(payload)


def loaded_image_from_rgb_array(
    path: str | Path,
    rgb_array: np.ndarray,
    *,
    rgba_image: Image.Image | None = None,
) -> LoadedImage:
    image_path = Path(path).resolve()
    normalized_rgb = _normalize_rgb_array(rgb_array)

    if rgba_image is None:
        rgba_image = Image.fromarray(normalized_rgb, mode="RGB").convert("RGBA")

    qimage = _pil_rgba_to_qimage(rgba_image)
    pixmap = QPixmap.fromImage(qimage)

    return LoadedImage(
        path=image_path,
        qimage=qimage,
        pixmap=pixmap,
        rgb_array=normalized_rgb,
        width=qimage.width(),
        height=qimage.height(),
    )


def _pil_rgba_to_qimage(image: Image.Image) -> QImage:
    data = image.tobytes("raw", "RGBA")
    qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
    return qimage.copy()


def _normalize_rgb_array(rgb_array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(rgb_array)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Ожидается RGB-массив изображения с тремя каналами.")
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return array.copy()
