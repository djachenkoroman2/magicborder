from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OIV_SYSTEM_NAME = "OIV"
OIV_SCALES_FILE_NAME = "oiv_ampelometric_scales.json"


class OivScaleError(ValueError):
    """Raised when the OIV scale JSON is missing required classifier data."""


@dataclass(frozen=True, slots=True)
class OivScaleInterval:
    score: int
    label: str
    min_value: float | None
    max_value: float | None
    include_min: bool = True
    include_max: bool = False
    comment: str = ""

    @classmethod
    def from_dict(cls, data: Any, *, trait_code: str) -> OivScaleInterval:
        if not isinstance(data, dict):
            raise OivScaleError(f"Интервал шкалы {trait_code} должен быть объектом.")

        score = data.get("score")
        if not isinstance(score, int):
            raise OivScaleError(f"Интервал шкалы {trait_code} должен содержать целый score.")

        label = str(data.get("label") or "").strip()
        if not label:
            raise OivScaleError(f"Интервал шкалы {trait_code} должен содержать label.")

        min_value = _optional_float(data.get("min"), f"{trait_code}.scale.min")
        max_value = _optional_float(data.get("max"), f"{trait_code}.scale.max")
        if min_value is not None and max_value is not None and min_value >= max_value:
            raise OivScaleError(f"Интервал шкалы {trait_code} имеет некорректные границы.")

        return cls(
            score=score,
            label=label,
            min_value=min_value,
            max_value=max_value,
            include_min=bool(data.get("include_min", True)),
            include_max=bool(data.get("include_max", False)),
            comment=str(data.get("comment") or "").strip(),
        )

    def contains(self, value: float) -> bool:
        if self.min_value is not None:
            if self.include_min:
                if value < self.min_value:
                    return False
            elif value <= self.min_value:
                return False

        if self.max_value is not None:
            if self.include_max:
                if value > self.max_value:
                    return False
            elif value >= self.max_value:
                return False

        return True


@dataclass(frozen=True, slots=True)
class OivTrait:
    code: str
    short_name: str
    official_name: str
    measurement_kind: str
    tool_kinds: tuple[str, ...]
    unit: str
    scale: tuple[OivScaleInterval, ...]
    boundary_method: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> OivTrait:
        if not isinstance(data, dict):
            raise OivScaleError("OIV-признак должен быть объектом.")

        code = normalize_oiv_code(data.get("code"))
        if not code:
            raise OivScaleError("OIV-признак должен содержать code.")

        short_name = str(data.get("short_name") or "").strip()
        official_name = str(data.get("official_name") or "").strip()
        measurement_kind = str(data.get("measurement_kind") or "").strip()
        unit = str(data.get("unit") or "").strip()
        if not short_name or not official_name or not measurement_kind or not unit:
            raise OivScaleError(f"Признак {code} содержит неполное описание.")

        raw_tool_kinds = data.get("tool_kinds")
        if not isinstance(raw_tool_kinds, list) or not raw_tool_kinds:
            raise OivScaleError(f"Признак {code} должен содержать непустой tool_kinds.")
        tool_kinds = tuple(str(item).strip() for item in raw_tool_kinds if str(item).strip())
        if not tool_kinds:
            raise OivScaleError(f"Признак {code} должен содержать непустой tool_kinds.")

        raw_scale = data.get("scale")
        if not isinstance(raw_scale, list) or not raw_scale:
            raise OivScaleError(f"Признак {code} должен содержать непустую scale.")
        scale = tuple(OivScaleInterval.from_dict(item, trait_code=code) for item in raw_scale)
        _validate_scale_ranges(code, scale)

        return cls(
            code=code,
            short_name=short_name,
            official_name=official_name,
            measurement_kind=measurement_kind,
            tool_kinds=tool_kinds,
            unit=unit,
            scale=scale,
            boundary_method=str(data.get("boundary_method") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class OivClassificationResult:
    status: str
    message: str
    code: str
    value: float | None = None
    unit: str = ""
    score: int | None = None
    label: str = ""
    comment: str = ""
    trait: OivTrait | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class OivScaleCatalog:
    def __init__(self, traits: list[OivTrait]) -> None:
        self._traits = tuple(traits)
        self._traits_by_code = {trait.code: trait for trait in self._traits}
        if len(self._traits_by_code) != len(self._traits):
            raise OivScaleError("Коды OIV в справочнике должны быть уникальными.")

    @property
    def traits(self) -> tuple[OivTrait, ...]:
        return self._traits

    def traits_for_tool(self, tool_kind: str) -> list[OivTrait]:
        normalized_tool_kind = str(tool_kind or "").strip()
        return [
            trait
            for trait in self._traits
            if normalized_tool_kind in trait.tool_kinds
        ]

    def trait_by_code(self, code: str | None) -> OivTrait | None:
        normalized_code = normalize_oiv_code(code)
        if not normalized_code:
            return None
        return self._traits_by_code.get(normalized_code)

    def classify(
        self,
        code: str | None,
        value: float,
        *,
        tool_kind: str | None = None,
    ) -> OivClassificationResult:
        normalized_code = normalize_oiv_code(code)
        if not normalized_code:
            return OivClassificationResult(
                status="not_selected",
                message="Не выбрано",
                code="",
            )

        trait = self.trait_by_code(normalized_code)
        if trait is None:
            return OivClassificationResult(
                status="unknown_trait",
                message="Неизвестный признак",
                code=normalized_code,
                value=value,
            )

        normalized_tool_kind = str(tool_kind or "").strip()
        if normalized_tool_kind and normalized_tool_kind not in trait.tool_kinds:
            return OivClassificationResult(
                status="incompatible_tool",
                message="Несовместимый признак",
                code=trait.code,
                value=value,
                unit=trait.unit,
                trait=trait,
            )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = math.nan
        if not math.isfinite(numeric_value):
            return OivClassificationResult(
                status="invalid_value",
                message="-",
                code=trait.code,
                value=None,
                unit=trait.unit,
                trait=trait,
            )

        for interval in trait.scale:
            if interval.contains(numeric_value):
                return OivClassificationResult(
                    status="ok",
                    message="ok",
                    code=trait.code,
                    value=numeric_value,
                    unit=trait.unit,
                    score=interval.score,
                    label=interval.label,
                    comment=interval.comment,
                    trait=trait,
                )

        return OivClassificationResult(
            status="no_match",
            message="Нет оценки по шкале",
            code=trait.code,
            value=numeric_value,
            unit=trait.unit,
            trait=trait,
        )


_DEFAULT_CATALOG: OivScaleCatalog | None = None


def default_scales_path() -> Path:
    module_path = Path(__file__).resolve()
    candidates = [
        module_path.parent / OIV_SCALES_FILE_NAME,
        module_path.parents[1] / OIV_SCALES_FILE_NAME,
        module_path.parents[2] / OIV_SCALES_FILE_NAME,
        Path.cwd() / OIV_SCALES_FILE_NAME,
    ]
    pyinstaller_root = getattr(sys, "_MEIPASS", None)
    if pyinstaller_root:
        bundle_root = Path(pyinstaller_root)
        candidates.extend(
            [
                bundle_root / OIV_SCALES_FILE_NAME,
                bundle_root / "magicborder" / OIV_SCALES_FILE_NAME,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[2]


def load_oiv_catalog(path: str | Path | None = None) -> OivScaleCatalog:
    scales_path = Path(path) if path is not None else default_scales_path()
    try:
        payload = json.loads(scales_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OivScaleError(f"Файл шкал OIV не найден: {scales_path}") from exc
    except json.JSONDecodeError as exc:
        raise OivScaleError(f"Некорректный JSON шкал OIV: {exc}") from exc

    if not isinstance(payload, dict):
        raise OivScaleError("JSON шкал OIV должен содержать объект верхнего уровня.")
    traits_payload = payload.get("traits")
    if not isinstance(traits_payload, list) or not traits_payload:
        raise OivScaleError("JSON шкал OIV должен содержать непустой список traits.")

    traits = [OivTrait.from_dict(item) for item in traits_payload]
    if any(trait.code == "OIV 616" for trait in traits):
        raise OivScaleError("OIV 616 не должен быть доступен как ампелометрическая шкала.")
    return OivScaleCatalog(traits)


def default_oiv_catalog() -> OivScaleCatalog:
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        _DEFAULT_CATALOG = load_oiv_catalog()
    return _DEFAULT_CATALOG


def normalize_oiv_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    compact = text.replace("_", " ").replace("-", " ")
    parts = compact.split()
    if len(parts) == 1 and parts[0].startswith("OIV"):
        suffix = parts[0][3:].strip()
        if suffix.isdigit():
            return f"OIV {int(suffix):03d}"
    if len(parts) == 1 and parts[0].isdigit():
        return f"OIV {int(parts[0]):03d}"
    if len(parts) == 2 and parts[0] == "OIV" and parts[1].isdigit():
        return f"OIV {int(parts[1]):03d}"
    return " ".join(parts)


def make_status_result(
    *,
    status: str,
    message: str,
    code: str | None = None,
    value: float | None = None,
    unit: str = "",
    trait: OivTrait | None = None,
) -> OivClassificationResult:
    normalized_code = normalize_oiv_code(code) if code is not None else (trait.code if trait else "")
    return OivClassificationResult(
        status=status,
        message=message,
        code=normalized_code,
        value=value,
        unit=unit,
        trait=trait,
    )


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OivScaleError(f"Поле {field_name} должно быть числом или null.") from exc
    if not math.isfinite(number):
        raise OivScaleError(f"Поле {field_name} должно быть конечным числом.")
    return number


def _validate_scale_ranges(code: str, scale: tuple[OivScaleInterval, ...]) -> None:
    if scale[0].min_value is not None:
        raise OivScaleError(f"Шкала {code} должна начинаться открытым нижним интервалом.")
    if scale[-1].max_value is not None:
        raise OivScaleError(f"Шкала {code} должна завершаться открытым верхним интервалом.")

    previous_max: float | None = None
    previous_include_max = False
    for index, interval in enumerate(scale):
        if index == 0:
            previous_max = interval.max_value
            previous_include_max = interval.include_max
            continue
        if interval.min_value != previous_max:
            raise OivScaleError(f"Шкала {code} содержит разрыв или пересечение интервалов.")
        if previous_include_max and interval.include_min:
            raise OivScaleError(f"Шкала {code} включает общую границу дважды.")
        previous_max = interval.max_value
        previous_include_max = interval.include_max
