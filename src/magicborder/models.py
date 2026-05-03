from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _require_number(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Поле '{field_name}' должно быть числом.") from exc


def _require_int(value: Any, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Поле '{field_name}' должно быть целым числом.") from exc
    if number <= 0:
        raise ValueError(f"Поле '{field_name}' должно быть положительным.")
    return number


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": round(float(self.x), 3), "y": round(float(self.y), 3)}

    @classmethod
    def from_dict(cls, data: Any) -> "Point":
        if not isinstance(data, dict):
            raise ValueError("Каждая точка контура должна быть объектом с полями 'x' и 'y'.")
        return cls(
            x=_require_number(data.get("x"), "x"),
            y=_require_number(data.get("y"), "y"),
        )


@dataclass(slots=True)
class Annotation:
    image_path: str
    image_width: int
    image_height: int
    points: list[Point]
    closed: bool = True
    version: int = 1

    def __post_init__(self) -> None:
        self.image_width = _require_int(self.image_width, "image_width")
        self.image_height = _require_int(self.image_height, "image_height")
        self.image_path = str(self.image_path or "")
        self.closed = bool(self.closed)
        self.version = max(1, int(self.version))
        if len(self.points) < 3:
            raise ValueError("Контур должен содержать минимум 3 точки.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "image_path": self.image_path,
            "image_size": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "closed": self.closed,
            "points": [point.to_dict() for point in self.points],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "Annotation":
        if not isinstance(data, dict):
            raise ValueError("JSON-аннотация должна содержать объект верхнего уровня.")

        image_size = data.get("image_size", {})
        if not isinstance(image_size, dict):
            image_size = {}

        image_width = image_size.get("width", data.get("image_width"))
        image_height = image_size.get("height", data.get("image_height"))
        raw_points = data.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("Поле 'points' должно содержать список точек.")

        return cls(
            image_path=str(data.get("image_path", "")),
            image_width=_require_int(image_width, "image_width"),
            image_height=_require_int(image_height, "image_height"),
            closed=bool(data.get("closed", True)),
            version=int(data.get("version", 1)),
            points=[Point.from_dict(item) for item in raw_points],
        )


def _optional_positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _normalize_project_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip("/")


PROJECT_IMAGE_METADATA_DEFAULTS: dict[str, str] = {
    "sample_id": "",
    "added_at": "",
    "captured_at": "",
    "illumination": "",
    "humidity": "",
    "wind_speed": "",
    "wind_direction": "",
    "latitude": "",
    "longitude": "",
    "diagnosis": "Не указано",
    "notes": "",
}


def default_project_image_metadata(
    *,
    sample_id: str = "",
    added_at: str = "",
    captured_at: str = "",
) -> dict[str, str]:
    metadata = dict(PROJECT_IMAGE_METADATA_DEFAULTS)
    metadata["sample_id"] = str(sample_id or "")
    metadata["added_at"] = str(added_at or "")
    metadata["captured_at"] = str(captured_at or "")
    return metadata


@dataclass(slots=True)
class ProjectImageRecord:
    id: str
    relative_path: str
    display_name: str
    image_width: int | None = None
    image_height: int | None = None
    annotation: Annotation | None = None
    annotation_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=default_project_image_metadata)
    raw_annotation: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip()
        self.relative_path = _normalize_project_path(self.relative_path)
        self.display_name = str(self.display_name or self.relative_path.rsplit("/", 1)[-1])
        self.image_width = _optional_positive_int(self.image_width)
        self.image_height = _optional_positive_int(self.image_height)
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        normalized_metadata: dict[str, Any] = default_project_image_metadata()
        normalized_metadata.update(self.metadata)
        self.metadata = normalized_metadata

        if not self.id:
            raise ValueError("Запись изображения должна содержать id.")
        if not self.relative_path:
            raise ValueError("Запись изображения должна содержать относительный путь.")

    def has_annotation(self) -> bool:
        return self.annotation is not None

    def point_count(self) -> int:
        if self.annotation is None:
            return 0
        return len(self.annotation.points)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "path": self.relative_path,
            "display_name": self.display_name,
            "image_size": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "annotation": None,
            "metadata": dict(self.metadata),
        }
        if self.annotation is not None:
            payload["annotation"] = self.annotation.to_dict()
        elif self.raw_annotation is not None:
            payload["annotation"] = self.raw_annotation
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> "ProjectImageRecord":
        if not isinstance(data, dict):
            raise ValueError("Запись изображения в проекте должна быть объектом.")

        relative_path = _normalize_project_path(data.get("path", data.get("relative_path", "")))
        image_size = data.get("image_size", {})
        if not isinstance(image_size, dict):
            image_size = {}

        annotation_payload = data.get("annotation")
        annotation = None
        annotation_error = None
        raw_annotation = None
        if annotation_payload is not None:
            raw_annotation = annotation_payload
            try:
                annotation = Annotation.from_dict(annotation_payload)
                raw_annotation = None
            except ValueError as exc:
                annotation_error = str(exc)

        record_id = str(data.get("id") or relative_path)
        display_name = str(data.get("display_name") or relative_path.rsplit("/", 1)[-1])
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        migrated_metadata = dict(metadata)
        for key in PROJECT_IMAGE_METADATA_DEFAULTS:
            if key in data and key not in migrated_metadata:
                migrated_metadata[key] = data[key]
        return cls(
            id=record_id,
            relative_path=relative_path,
            display_name=display_name,
            image_width=_optional_positive_int(image_size.get("width", data.get("image_width"))),
            image_height=_optional_positive_int(image_size.get("height", data.get("image_height"))),
            annotation=annotation,
            annotation_error=annotation_error,
            metadata=migrated_metadata,
            raw_annotation=raw_annotation,
        )


@dataclass(slots=True)
class ProjectDocument:
    name: str
    images: list[ProjectImageRecord]
    version: int = 1
    images_dir: str = "images"

    def __post_init__(self) -> None:
        self.name = str(self.name or "project").strip() or "project"
        self.version = _optional_positive_int(self.version) or 1
        self.images_dir = _normalize_project_path(self.images_dir or "images") or "images"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "images_dir": self.images_dir,
            "images": [record.to_dict() for record in self.images],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ProjectDocument":
        if not isinstance(data, dict):
            raise ValueError("JSON проекта должен содержать объект верхнего уровня.")

        raw_images = data.get("images", [])
        if not isinstance(raw_images, list):
            raise ValueError("Поле 'images' должно содержать список изображений.")

        images: list[ProjectImageRecord] = []
        for item in raw_images:
            try:
                images.append(ProjectImageRecord.from_dict(item))
            except ValueError:
                continue

        return cls(
            name=str(data.get("name") or data.get("project_name") or "project"),
            version=_optional_positive_int(data.get("version")) or 1,
            images_dir=str(data.get("images_dir") or "images"),
            images=images,
        )
