from __future__ import annotations

from typing import Any

import pytest

from magicborder.models import (
    ANGLE_LINE_COLOR,
    CONTOUR_LINE_COLOR,
    PROJECT_FORMAT_VERSION,
    Annotation,
    ImageCalibration,
    Point,
    ProjectAngleMeasurement,
    ProjectDocument,
    ProjectImageFileInfo,
    ProjectImageMeasurements,
    ProjectImageRecord,
    ProjectMeasurementAssessment,
    _normalize_project_path,
    _optional_positive_int,
    normalize_line_color,
)


def _points() -> list[Point]:
    return [Point(0, 0), Point(10, 0), Point(10, 10)]


class TestNormalizeLineColor:
    def test_valid_hex_is_lowercased(self) -> None:
        assert normalize_line_color("#AABBCC", CONTOUR_LINE_COLOR) == "#aabbcc"

    def test_three_digit_hex_is_rejected(self) -> None:
        assert normalize_line_color("#abc", CONTOUR_LINE_COLOR) == CONTOUR_LINE_COLOR

    def test_invalid_default_falls_back_to_black(self) -> None:
        assert normalize_line_color("не цвет", "тоже не цвет") == "#000000"

    @pytest.mark.parametrize("value", [None, "", 123, [], {"a": 1}])
    def test_non_string_input_uses_default(self, value: Any) -> None:
        assert normalize_line_color(value, ANGLE_LINE_COLOR) == ANGLE_LINE_COLOR

    def test_default_is_lowercased_too(self) -> None:
        assert normalize_line_color(None, "#AABBCC") == "#aabbcc"


class TestPoint:
    @pytest.mark.parametrize("payload", ["строка", 5, [], None])
    def test_from_dict_requires_object(self, payload: Any) -> None:
        with pytest.raises(ValueError, match="должна быть объектом"):
            Point.from_dict(payload)

    def test_from_dict_requires_x(self) -> None:
        with pytest.raises(ValueError, match="'x'"):
            Point.from_dict({"y": 1})

    def test_from_dict_rejects_non_numeric(self) -> None:
        with pytest.raises(ValueError, match="'y'"):
            Point.from_dict({"x": 1, "y": "около десяти"})

    def test_to_dict_rounds_to_three_decimals(self) -> None:
        assert Point(1.234567, 2.987654).to_dict() == {"x": 1.235, "y": 2.988}


class TestImageCalibration:
    @pytest.mark.parametrize("length_mm", [0, -5, -0.001])
    def test_non_positive_length_is_rejected(self, length_mm: float) -> None:
        with pytest.raises(ValueError, match="положительной"):
            ImageCalibration(start=Point(0, 0), end=Point(10, 0), length_mm=length_mm)

    def test_zero_pixel_length_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ненулевую длину"):
            ImageCalibration(start=Point(3, 4), end=Point(3, 4), length_mm=10)

    def test_non_numeric_length_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="'length_mm'"):
            ImageCalibration(start=Point(0, 0), end=Point(10, 0), length_mm="десять")

    def test_scale_arithmetic(self) -> None:
        calibration = ImageCalibration(
            start=Point(0, 0), end=Point(3, 4), length_mm=2.5
        )

        assert calibration.pixel_length() == pytest.approx(5.0)
        assert calibration.pixels_per_mm() == pytest.approx(2.0)
        assert calibration.mm_per_pixel() == pytest.approx(0.5)

    @pytest.mark.parametrize("payload", ["строка", 5, [], None])
    def test_from_dict_requires_object(self, payload: Any) -> None:
        with pytest.raises(ValueError, match="должна быть объектом"):
            ImageCalibration.from_dict(payload)

    def test_dict_points_are_converted(self) -> None:
        calibration = ImageCalibration(
            start={"x": 0, "y": 0},
            end={"x": 0, "y": 8},
            length_mm=4,
        )

        assert calibration.start == Point(0.0, 0.0)
        assert calibration.pixels_per_mm() == pytest.approx(2.0)


class TestAnnotation:
    def test_less_than_three_points_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="минимум 3 точки"):
            Annotation(
                image_path="a.png",
                image_width=10,
                image_height=10,
                points=[Point(0, 0), Point(1, 1)],
            )

    @pytest.mark.parametrize("width", [0, -3, "широкое", None])
    def test_invalid_image_width(self, width: Any) -> None:
        with pytest.raises(ValueError, match="'image_width'"):
            Annotation(
                image_path="a.png", image_width=width, image_height=10, points=_points()
            )

    @pytest.mark.parametrize("height", [0, -3, "высокое", None])
    def test_invalid_image_height(self, height: Any) -> None:
        with pytest.raises(ValueError, match="'image_height'"):
            Annotation(
                image_path="a.png",
                image_width=10,
                image_height=height,
                points=_points(),
            )

    def test_legacy_flat_image_dimensions(self) -> None:
        annotation = Annotation.from_dict(
            {
                "image_path": "a.png",
                "image_width": 40,
                "image_height": 30,
                "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
            }
        )

        assert (annotation.image_width, annotation.image_height) == (40, 30)

    def test_image_size_takes_priority_over_legacy_fields(self) -> None:
        annotation = Annotation.from_dict(
            {
                "image_size": {"width": 40, "height": 30},
                "image_width": 4,
                "image_height": 3,
                "points": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
            }
        )

        assert (annotation.image_width, annotation.image_height) == (40, 30)

    @pytest.mark.parametrize("points", [None, "точки", {"x": 1}, 5])
    def test_points_must_be_a_list(self, points: Any) -> None:
        payload: dict[str, Any] = {"image_size": {"width": 4, "height": 4}}
        if points is not None:
            payload["points"] = points

        with pytest.raises(ValueError, match="'points'"):
            Annotation.from_dict(payload)

    @pytest.mark.parametrize("payload", ["строка", 5, [], None])
    def test_from_dict_requires_object(self, payload: Any) -> None:
        with pytest.raises(ValueError, match="объект верхнего уровня"):
            Annotation.from_dict(payload)


class TestProjectImageFileInfo:
    def test_missing_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="должна содержать id"):
            ProjectImageFileInfo(id="  ", path="images/a.png", display_name="")

    @pytest.mark.parametrize("path", ["", None, "/", "///"])
    def test_missing_path_is_rejected(self, path: Any) -> None:
        with pytest.raises(ValueError, match="относительный путь"):
            ProjectImageFileInfo(id="a", path=path, display_name="")

    def test_whitespace_only_path_is_currently_accepted(self) -> None:
        # _normalize_project_path обрезает только слеши, поэтому пробельный путь
        # проходит проверку. Тест фиксирует текущее поведение.
        info = ProjectImageFileInfo(id="a", path="   ", display_name="")

        assert info.path == "   "

    def test_display_name_defaults_to_last_path_segment(self) -> None:
        info = ProjectImageFileInfo(id="a", path="images/сад/лист.png", display_name="")

        assert info.display_name == "лист.png"

    def test_from_dict_requires_object(self) -> None:
        with pytest.raises(ValueError, match="'file'"):
            ProjectImageFileInfo.from_dict("строка")


class TestNormalizeProjectPath:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("images\\leaf.png", "images/leaf.png"),
            ("/images/leaf.png/", "images/leaf.png"),
            ("//images//leaf.png//", "images//leaf.png"),
            ("\\images\\leaf.png\\", "images/leaf.png"),
            (None, ""),
            ("", ""),
        ],
    )
    def test_normalisation(self, value: Any, expected: str) -> None:
        assert _normalize_project_path(value) == expected


class TestOptionalPositiveInt:
    @pytest.mark.parametrize("value", [0, -5, "abc", "", None, [], {}])
    def test_invalid_values_become_none(self, value: Any) -> None:
        assert _optional_positive_int(value) is None

    @pytest.mark.parametrize(("value", "expected"), [(3, 3), ("7", 7), (4.9, 4)])
    def test_valid_values(self, value: Any, expected: int) -> None:
        assert _optional_positive_int(value) == expected


class TestProjectImageRecordMetadata:
    def test_unknown_metadata_keys_survive_round_trip(self) -> None:
        record = ProjectImageRecord(
            id="a",
            relative_path="images/a.png",
            metadata={"diagnosis": "здоров", "сорт": "Каберне"},
        )

        payload = record.to_dict()
        assert payload["metadata_extra"] == {"сорт": "Каберне"}

        restored = ProjectImageRecord.from_dict(payload)
        assert restored.metadata["сорт"] == "Каберне"
        assert restored.metadata["diagnosis"] == "здоров"

    def test_set_metadata_value_stores_unknown_key_in_extra(self) -> None:
        record = ProjectImageRecord(id="a", relative_path="images/a.png")

        record.set_metadata_value("сорт", "Мерло")

        assert record.extra_groups["metadata_extra"] == {"сорт": "Мерло"}
        assert record.metadata["сорт"] == "Мерло"

    def test_set_metadata_value_repairs_broken_extra_container(self) -> None:
        record = ProjectImageRecord(id="a", relative_path="images/a.png")
        record.extra_groups["metadata_extra"] = "не словарь"

        record.set_metadata_value("сорт", "Мерло")

        assert record.extra_groups["metadata_extra"] == {"сорт": "Мерло"}

    def test_metadata_setter_clears_extra_when_no_unknown_keys(self) -> None:
        record = ProjectImageRecord(
            id="a",
            relative_path="images/a.png",
            metadata={"сорт": "Каберне"},
        )
        assert "metadata_extra" in record.extra_groups

        record.metadata = {"diagnosis": "здоров"}

        assert "metadata_extra" not in record.extra_groups
        assert record.metadata["diagnosis"] == "здоров"

    def test_metadata_setter_ignores_non_dict(self) -> None:
        record = ProjectImageRecord(id="a", relative_path="images/a.png")

        record.metadata = "не словарь"

        assert record.metadata["diagnosis"] == "Не указано"

    def test_known_metadata_keys_are_routed_to_groups(self) -> None:
        record = ProjectImageRecord(id="a", relative_path="images/a.png")

        record.set_metadata_value("added_at", "2026-08-31T00:00:00")
        record.set_metadata_value("humidity", "70%")
        record.set_metadata_value("diagnosis", "мучнистая роса")
        record.set_metadata_value("notes", "две строки")

        assert record.file.added_at == "2026-08-31T00:00:00"
        assert record.location.humidity == "70%"
        assert record.details.diagnosis == "мучнистая роса"
        assert record.details.notes == "две строки"
        assert "metadata_extra" not in record.extra_groups


class TestProjectImageRecordExtraGroups:
    def test_unknown_sections_survive_round_trip(self) -> None:
        payload = {
            "file": {"id": "a", "path": "images/a.png", "display_name": "a.png"},
            "contour": {"annotation": None},
            "location": {},
            "details": {},
            "будущая_секция": {"ключ": "значение"},
        }

        record = ProjectImageRecord.from_dict(payload)

        assert record.extra_groups["будущая_секция"] == {"ключ": "значение"}
        assert record.to_dict()["будущая_секция"] == {"ключ": "значение"}

    def test_from_dict_requires_object(self) -> None:
        with pytest.raises(ValueError, match="должна быть объектом"):
            ProjectImageRecord.from_dict("строка")


class TestProjectImageMeasurements:
    def test_empty_measurements_have_no_data(self) -> None:
        assert ProjectImageMeasurements().has_data() is False

    def test_extra_groups_survive_round_trip(self) -> None:
        measurements = ProjectImageMeasurements.from_dict(
            {"angles": [], "segments": [], "areas": [{"id": "area-1"}]}
        )

        assert measurements.has_data() is True
        assert measurements.extra_groups == {"areas": [{"id": "area-1"}]}
        assert measurements.to_dict()["areas"] == [{"id": "area-1"}]

    def test_angles_only_counts_as_data(self) -> None:
        measurements = ProjectImageMeasurements(
            angles=[
                ProjectAngleMeasurement(
                    id="angle-a",
                    first=Point(0, 1),
                    vertex=Point(0, 0),
                    second=Point(1, 0),
                )
            ]
        )

        assert measurements.has_data() is True

    def test_non_dict_payload_is_tolerated(self) -> None:
        measurements = ProjectImageMeasurements.from_dict("не словарь")

        assert measurements.has_data() is False


class TestProjectDocument:
    def test_images_must_be_a_list(self) -> None:
        with pytest.raises(ValueError, match="'images'"):
            ProjectDocument.from_dict({"images": "нет"})

    def test_from_dict_requires_object(self) -> None:
        with pytest.raises(ValueError, match="объект верхнего уровня"):
            ProjectDocument.from_dict("строка")

    def test_name_falls_back_to_project_name(self) -> None:
        document = ProjectDocument.from_dict({"project_name": "старое имя"})

        assert document.name == "старое имя"

    def test_name_falls_back_to_default(self) -> None:
        assert ProjectDocument.from_dict({}).name == "project"
        assert ProjectDocument.from_dict({"name": "   "}).name == "project"

    @pytest.mark.parametrize("version", [0, -1, "abc", None])
    def test_non_positive_version_uses_format_version(self, version: Any) -> None:
        assert (
            ProjectDocument(name="p", images=[], version=version).version
            == PROJECT_FORMAT_VERSION
        )
        assert (
            ProjectDocument.from_dict({"version": version}).version
            == PROJECT_FORMAT_VERSION
        )

    @pytest.mark.parametrize("images_dir", ["", "/", "//", None])
    def test_blank_images_dir_falls_back(self, images_dir: Any) -> None:
        assert (
            ProjectDocument(name="p", images=[], images_dir=images_dir).images_dir
            == "images"
        )

    def test_broken_image_records_are_skipped(self) -> None:
        document = ProjectDocument.from_dict(
            {
                "images": [
                    "мусор",
                    {"file": {"id": "a", "path": "images/a.png"}},
                ]
            }
        )

        assert [record.id for record in document.images] == ["a"]


class TestProjectMeasurementAssessment:
    @pytest.mark.parametrize("payload", ["строка", 5, [], None])
    def test_non_object_becomes_none(self, payload: Any) -> None:
        assert ProjectMeasurementAssessment.from_dict(payload) is None

    @pytest.mark.parametrize("code", [None, "", "   "])
    def test_blank_code_becomes_none(self, code: Any) -> None:
        assert ProjectMeasurementAssessment.from_dict({"code": code}) is None

    def test_system_defaults_to_oiv(self) -> None:
        assert (
            ProjectMeasurementAssessment.from_dict(
                {"code": "OIV 601", "system": ""}
            ).system
            == "OIV"
        )
        assert ProjectMeasurementAssessment(system="  ", code="OIV 601").system == "OIV"

    def test_values_are_trimmed(self) -> None:
        assessment = ProjectMeasurementAssessment.from_dict(
            {"code": "  OIV 601  ", "system": " OIV "}
        )

        assert assessment is not None
        assert assessment.code == "OIV 601"
