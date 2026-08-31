from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from magicborder import oiv
from magicborder.oiv import (
    OIV_SCALES_FILE_NAME,
    OivScaleCatalog,
    OivScaleError,
    OivScaleInterval,
    OivTrait,
    _optional_float,
    default_oiv_catalog,
    default_scales_path,
    load_oiv_catalog,
    make_status_result,
    normalize_oiv_code,
)


def _scale() -> list[dict[str, Any]]:
    return [
        {"score": 1, "label": "низкий", "min": None, "max": 10.0},
        {"score": 3, "label": "средний", "min": 10.0, "max": 20.0},
        {"score": 5, "label": "высокий", "min": 20.0, "max": None},
    ]


def _trait_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": "OIV 601",
        "short_name": "Длина",
        "official_name": "Длина главной жилки",
        "measurement_kind": "length",
        "tool_kinds": ["segment"],
        "unit": "мм",
        "scale": _scale(),
    }
    payload.update(overrides)
    return payload


def _catalog(*traits: dict[str, Any]) -> OivScaleCatalog:
    return OivScaleCatalog([OivTrait.from_dict(item) for item in (traits or (_trait_payload(),))])


def _write_catalog(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class TestLoadOivCatalog:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(OivScaleError, match="Файл шкал OIV не найден"):
            load_oiv_catalog(tmp_path / "нет.json")

    def test_broken_json(self, tmp_path: Path) -> None:
        path = tmp_path / "scales.json"
        path.write_text("{ сломано", encoding="utf-8")

        with pytest.raises(OivScaleError, match="Некорректный JSON шкал OIV"):
            load_oiv_catalog(path)

    @pytest.mark.parametrize("payload", [[], "строка", 42])
    def test_top_level_must_be_object(self, tmp_path: Path, payload: Any) -> None:
        path = _write_catalog(tmp_path / "scales.json", payload)

        with pytest.raises(OivScaleError, match="объект верхнего уровня"):
            load_oiv_catalog(path)

    @pytest.mark.parametrize("traits", [None, [], {}, "нет"])
    def test_traits_must_be_non_empty_list(self, tmp_path: Path, traits: Any) -> None:
        payload = {} if traits is None else {"traits": traits}
        path = _write_catalog(tmp_path / "scales.json", payload)

        with pytest.raises(OivScaleError, match="непустой список traits"):
            load_oiv_catalog(path)

    def test_oiv_616_is_rejected(self, tmp_path: Path) -> None:
        path = _write_catalog(
            tmp_path / "scales.json",
            {"traits": [_trait_payload(code="OIV 616")]},
        )

        with pytest.raises(OivScaleError, match="OIV 616"):
            load_oiv_catalog(path)

    def test_valid_catalog_is_loaded(self, tmp_path: Path) -> None:
        path = _write_catalog(tmp_path / "scales.json", {"traits": [_trait_payload()]})

        catalog = load_oiv_catalog(path)

        assert [trait.code for trait in catalog.traits] == ["OIV 601"]


class TestOivScaleCatalog:
    def test_duplicate_codes_are_rejected(self) -> None:
        traits = [OivTrait.from_dict(_trait_payload()), OivTrait.from_dict(_trait_payload())]

        with pytest.raises(OivScaleError, match="уникальными"):
            OivScaleCatalog(traits)

    def test_default_catalog_is_cached(self) -> None:
        assert default_oiv_catalog() is default_oiv_catalog()


class TestDefaultScalesPath:
    def test_repository_copy_is_found(self) -> None:
        assert default_scales_path().name == OIV_SCALES_FILE_NAME
        assert default_scales_path().exists()

    def test_pyinstaller_bundle_root(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        empty_root = tmp_path / "a" / "b" / "c" / "package"
        empty_root.mkdir(parents=True)
        bundle_root = tmp_path / "bundle"
        bundle_root.mkdir()
        bundled_file = bundle_root / OIV_SCALES_FILE_NAME
        bundled_file.write_text("{}", encoding="utf-8")

        monkeypatch.chdir(tmp_path / "a")
        monkeypatch.setattr(oiv, "__file__", str(empty_root / "oiv.py"))
        monkeypatch.setattr(sys, "_MEIPASS", str(bundle_root), raising=False)

        assert default_scales_path() == bundled_file

    def test_falls_back_to_third_candidate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        empty_root = tmp_path / "a" / "b" / "c" / "package"
        empty_root.mkdir(parents=True)

        monkeypatch.chdir(tmp_path / "a")
        monkeypatch.setattr(oiv, "__file__", str(empty_root / "oiv.py"))
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)

        assert default_scales_path() == tmp_path / "a" / "b" / OIV_SCALES_FILE_NAME


class TestOivTraitValidation:
    @pytest.mark.parametrize("payload", ["строка", 5, [], None])
    def test_trait_must_be_object(self, payload: Any) -> None:
        with pytest.raises(OivScaleError, match="должен быть объектом"):
            OivTrait.from_dict(payload)

    @pytest.mark.parametrize("code", [None, "", "   "])
    def test_missing_code(self, code: Any) -> None:
        with pytest.raises(OivScaleError, match="должен содержать code"):
            OivTrait.from_dict(_trait_payload(code=code))

    @pytest.mark.parametrize(
        "field",
        ["short_name", "official_name", "measurement_kind", "unit"],
    )
    def test_empty_description_fields(self, field: str) -> None:
        with pytest.raises(OivScaleError, match="неполное описание"):
            OivTrait.from_dict(_trait_payload(**{field: "   "}))

    @pytest.mark.parametrize("tool_kinds", [None, [], "segment", {}])
    def test_invalid_tool_kinds(self, tool_kinds: Any) -> None:
        payload = _trait_payload()
        if tool_kinds is None:
            payload.pop("tool_kinds")
        else:
            payload["tool_kinds"] = tool_kinds

        with pytest.raises(OivScaleError, match="непустой tool_kinds"):
            OivTrait.from_dict(payload)

    def test_blank_tool_kinds_are_rejected(self) -> None:
        with pytest.raises(OivScaleError, match="непустой tool_kinds"):
            OivTrait.from_dict(_trait_payload(tool_kinds=["  ", ""]))

    @pytest.mark.parametrize("scale", [None, [], "шкала", {}])
    def test_invalid_scale(self, scale: Any) -> None:
        payload = _trait_payload()
        if scale is None:
            payload.pop("scale")
        else:
            payload["scale"] = scale

        with pytest.raises(OivScaleError, match="непустую scale"):
            OivTrait.from_dict(payload)

    def test_valid_trait_is_normalised(self) -> None:
        trait = OivTrait.from_dict(_trait_payload(code="oiv-601", tool_kinds=[" segment ", ""]))

        assert trait.code == "OIV 601"
        assert trait.tool_kinds == ("segment",)
        assert len(trait.scale) == 3


class TestOivScaleIntervalValidation:
    @pytest.mark.parametrize("payload", ["строка", 5, [], None])
    def test_interval_must_be_object(self, payload: Any) -> None:
        with pytest.raises(OivScaleError, match="должен быть объектом"):
            OivScaleInterval.from_dict(payload, trait_code="OIV 601")

    @pytest.mark.parametrize("score", ["3", 3.0, None])
    def test_score_must_be_int(self, score: Any) -> None:
        with pytest.raises(OivScaleError, match="целый score"):
            OivScaleInterval.from_dict(
                {"score": score, "label": "низкий", "min": None, "max": 1.0},
                trait_code="OIV 601",
            )

    @pytest.mark.parametrize("label", [None, "", "   "])
    def test_label_is_required(self, label: Any) -> None:
        data: dict[str, Any] = {"score": 1, "min": None, "max": 1.0}
        if label is not None:
            data["label"] = label

        with pytest.raises(OivScaleError, match="должен содержать label"):
            OivScaleInterval.from_dict(data, trait_code="OIV 601")

    @pytest.mark.parametrize(("min_value", "max_value"), [(10.0, 10.0), (20.0, 10.0)])
    def test_min_must_be_below_max(self, min_value: float, max_value: float) -> None:
        with pytest.raises(OivScaleError, match="некорректные границы"):
            OivScaleInterval.from_dict(
                {"score": 1, "label": "l", "min": min_value, "max": max_value},
                trait_code="OIV 601",
            )


class TestOptionalFloat:
    def test_none_means_open_boundary(self) -> None:
        assert _optional_float(None, "OIV 601.scale.min") is None

    @pytest.mark.parametrize("value", ["abc", [], {}])
    def test_non_numeric_is_rejected(self, value: Any) -> None:
        with pytest.raises(OivScaleError, match="должно быть числом или null"):
            _optional_float(value, "OIV 601.scale.min")

    @pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
    def test_non_finite_is_rejected(self, value: float) -> None:
        with pytest.raises(OivScaleError, match="конечным числом"):
            _optional_float(value, "OIV 601.scale.min")

    def test_numeric_string_is_accepted(self) -> None:
        assert _optional_float("12.5", "OIV 601.scale.min") == 12.5


class TestValidateScaleRanges:
    def test_first_interval_must_be_open(self) -> None:
        scale = _scale()
        scale[0]["min"] = 0.0

        with pytest.raises(OivScaleError, match="открытым нижним интервалом"):
            OivTrait.from_dict(_trait_payload(scale=scale))

    def test_last_interval_must_be_open(self) -> None:
        scale = _scale()
        scale[-1]["max"] = 30.0

        with pytest.raises(OivScaleError, match="открытым верхним интервалом"):
            OivTrait.from_dict(_trait_payload(scale=scale))

    def test_gap_between_intervals(self) -> None:
        scale = _scale()
        scale[1]["min"] = 12.0

        with pytest.raises(OivScaleError, match="разрыв или пересечение"):
            OivTrait.from_dict(_trait_payload(scale=scale))

    def test_overlapping_intervals(self) -> None:
        scale = _scale()
        scale[1]["min"] = 8.0

        with pytest.raises(OivScaleError, match="разрыв или пересечение"):
            OivTrait.from_dict(_trait_payload(scale=scale))

    def test_shared_boundary_included_twice(self) -> None:
        scale = _scale()
        scale[0]["include_max"] = True
        scale[1]["include_min"] = True

        with pytest.raises(OivScaleError, match="общую границу дважды"):
            OivTrait.from_dict(_trait_payload(scale=scale))


class TestOivScaleIntervalContains:
    @pytest.mark.parametrize(
        ("include_min", "include_max", "expected"),
        [
            (True, False, (True, False)),
            (False, True, (False, True)),
            (True, True, (True, True)),
            (False, False, (False, False)),
        ],
    )
    def test_boundary_inclusion_combinations(
        self,
        include_min: bool,
        include_max: bool,
        expected: tuple[bool, bool],
    ) -> None:
        interval = OivScaleInterval(
            score=1,
            label="l",
            min_value=10.0,
            max_value=20.0,
            include_min=include_min,
            include_max=include_max,
        )

        assert (interval.contains(10.0), interval.contains(20.0)) == expected
        assert interval.contains(15.0) is True
        assert interval.contains(9.0) is False
        assert interval.contains(21.0) is False

    def test_open_lower_boundary(self) -> None:
        interval = OivScaleInterval(score=1, label="l", min_value=None, max_value=10.0)

        assert interval.contains(-1_000_000.0) is True
        assert interval.contains(10.0) is False

    def test_open_upper_boundary(self) -> None:
        interval = OivScaleInterval(score=1, label="l", min_value=10.0, max_value=None)

        assert interval.contains(1_000_000.0) is True
        assert interval.contains(9.99) is False


class TestClassify:
    @pytest.mark.parametrize("code", [None, "", "   "])
    def test_missing_code_is_not_selected(self, code: Any) -> None:
        result = _catalog().classify(code, 5.0)

        assert result.status == "not_selected"
        assert result.message == "Не выбрано"
        assert result.code == ""
        assert result.ok is False

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_value_is_invalid(self, value: float) -> None:
        result = _catalog().classify("OIV 601", value, tool_kind="segment")

        assert result.status == "invalid_value"
        assert result.value is None
        assert result.unit == "мм"

    def test_blank_tool_kind_skips_compatibility_check(self) -> None:
        result = _catalog().classify("OIV 601", 15.0, tool_kind="")

        assert result.status == "ok"
        assert result.score == 3

    def test_incompatible_tool_kind(self) -> None:
        result = _catalog().classify("OIV 601", 15.0, tool_kind="angle")

        assert result.status == "incompatible_tool"

    def test_value_outside_every_interval(self) -> None:
        scale = [
            {"score": 1, "label": "низкий", "min": None, "max": 10.0, "include_max": False},
            {"score": 3, "label": "высокий", "min": 10.0, "max": None, "include_min": False},
        ]
        catalog = _catalog(_trait_payload(scale=scale))

        result = catalog.classify("OIV 601", 10.0, tool_kind="segment")

        assert result.status == "no_match"
        assert result.message == "Нет оценки по шкале"
        assert result.score is None

    def test_successful_classification(self) -> None:
        result = _catalog().classify("oiv601", 25.0, tool_kind="segment")

        assert result.ok is True
        assert result.code == "OIV 601"
        assert result.score == 5
        assert result.label == "высокий"


class TestNormalizeOivCode:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("oiv601", "OIV 601"),
            ("OIV601", "OIV 601"),
            ("601", "OIV 601"),
            ("OIV-601", "OIV 601"),
            ("OIV_601", "OIV 601"),
            (" 61 ", "OIV 061"),
            ("OIV 61", "OIV 061"),
            ("oiv 601 доп", "OIV 601 ДОП"),
            ("непонятно", "НЕПОНЯТНО"),
            (None, ""),
            ("", ""),
            ("   ", ""),
        ],
    )
    def test_normalisation(self, value: Any, expected: str) -> None:
        assert normalize_oiv_code(value) == expected


class TestMakeStatusResult:
    def test_code_is_taken_from_trait_when_missing(self) -> None:
        trait = OivTrait.from_dict(_trait_payload())

        result = make_status_result(status="pending", message="ждём", trait=trait)

        assert result.code == "OIV 601"
        assert result.trait is trait

    def test_explicit_code_is_normalised(self) -> None:
        result = make_status_result(status="pending", message="ждём", code="oiv-602")

        assert result.code == "OIV 602"

    def test_without_code_and_trait(self) -> None:
        result = make_status_result(status="pending", message="ждём")

        assert result.code == ""


class TestTraitByCode:
    @pytest.mark.parametrize("code", [None, "", "   "])
    def test_blank_code_returns_none(self, code: Any) -> None:
        assert _catalog().trait_by_code(code) is None

    def test_unknown_code_returns_none(self) -> None:
        assert _catalog().trait_by_code("OIV 999") is None

    def test_known_code_is_normalised_before_lookup(self) -> None:
        trait = _catalog().trait_by_code("oiv-601")

        assert trait is not None
        assert trait.code == "OIV 601"


class TestClassifyNonNumericValue:
    @pytest.mark.parametrize("value", ["не число", None, [], {}])
    def test_non_numeric_value_is_treated_as_invalid(self, value: Any) -> None:
        result = _catalog().classify("OIV 601", value, tool_kind="segment")

        assert result.status == "invalid_value"
        assert result.value is None
