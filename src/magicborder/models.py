from __future__ import annotations

from dataclasses import dataclass
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
