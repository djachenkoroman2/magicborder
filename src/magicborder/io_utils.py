from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image, UnidentifiedImageError
from PyQt5.QtGui import QImage, QPixmap

from .models import Annotation, ProjectDocument

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


def save_project(path: str | Path, project: ProjectDocument) -> None:
    project_path = Path(path)
    project_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(project.to_dict(), ensure_ascii=False, indent=2)
    temp_path = project_path.with_name(f".{project_path.name}.tmp")
    temp_path.write_text(payload, encoding="utf-8")
    os.replace(temp_path, project_path)


def load_project(path: str | Path) -> ProjectDocument:
    project_path = Path(path)
    try:
        raw_text = project_path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Файл проекта не найден: {project_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON проекта: {exc}") from exc
    return ProjectDocument.from_dict(payload)


def read_image_captured_at(path: str | Path) -> str:
    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
    except (OSError, UnidentifiedImageError):
        return ""

    if not exif:
        return ""

    tag_by_name = {name: tag for tag, name in ExifTags.TAGS.items()}
    for tag_name in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        raw_value = exif.get(tag_by_name.get(tag_name))
        parsed_value = _parse_exif_datetime(raw_value)
        if parsed_value:
            return parsed_value
    return ""


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


def _parse_exif_datetime(value: object) -> str:
    if not value:
        return ""
    raw_text = str(value).strip()
    if not raw_text:
        return ""
    for format_string in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw_text, format_string).isoformat(timespec="seconds")
        except ValueError:
            continue
    return raw_text


def _normalize_rgb_array(rgb_array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(rgb_array)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Ожидается RGB-массив изображения с тремя каналами.")
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return array.copy()
