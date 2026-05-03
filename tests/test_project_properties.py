from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from magicborder.io_utils import load_project, save_project  # noqa: E402
from magicborder.main_window import MainWindow, _circle_contour_points, _qdatetime_from_text  # noqa: E402
from magicborder.models import Annotation, Point, ProjectDocument, ProjectImageRecord  # noqa: E402

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _APP = app
    return app


class ProjectPropertiesTest(unittest.TestCase):
    def test_circle_contour_points_are_centered_and_counted(self) -> None:
        points = _circle_contour_points(20, 20, 5)

        self.assertEqual(len(points), 5)
        self.assertAlmostEqual(points[0].x, 10.0)
        self.assertAlmostEqual(points[0].y, 3.0)
        self.assertTrue(all(0.0 <= point.x <= 20.0 for point in points))
        self.assertTrue(all(0.0 <= point.y <= 20.0 for point in points))

    def test_property_point_count_matches_selected_contour(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            annotation = Annotation(
                image_path="images/leaf.png",
                image_width=20,
                image_height=20,
                points=[
                    Point(2, 2),
                    Point(15, 2),
                    Point(17, 8),
                    Point(12, 17),
                    Point(3, 14),
                ],
            )
            project = ProjectDocument(
                name="check_points",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=20,
                        image_height=20,
                        annotation=annotation,
                    )
                ],
            )
            project_path = root / "check_points.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertEqual(window.property_points.text(), "5")

            window.canvas.set_contour(
                [
                    Point(1, 1),
                    Point(18, 1),
                    Point(18, 18),
                    Point(1, 18),
                ]
            )

            self.assertEqual(window.property_points.text(), "4")
            self.assertEqual(window.property_contour_pixels.text(), "324")
            self.assertEqual(len(window.project_document.images[0].annotation.points), 4)

            window.delete_current_contour()

            self.assertFalse(window.canvas.has_contour())
            self.assertIsNone(window.project_document.images[0].annotation)
            self.assertEqual(window.property_annotation.text(), "нет")
            self.assertEqual(window.property_points.text(), "-")
            self.assertEqual(window.property_contour_pixels.text(), "-")

    def test_new_contour_command_creates_project_annotation(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="new_contour",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=20,
                        image_height=20,
                    )
                ],
            )
            project_path = root / "new_contour.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertTrue(window.new_contour_action.isEnabled())
            with patch("magicborder.main_window.QInputDialog.getInt", return_value=(5, True)):
                window.create_new_contour()

            annotation = window.project_document.images[0].annotation
            self.assertIsNotNone(annotation)
            self.assertEqual(len(annotation.points), 5)
            self.assertEqual(window.property_points.text(), "5")

    def test_metadata_fields_save_and_file_name_renames_image(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="metadata",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=20,
                        image_height=20,
                    )
                ],
            )
            project_path = root / "metadata.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            window.metadata_humidity.setText("65")
            window._handle_metadata_line_edit_finished("humidity", window.metadata_humidity)
            window.metadata_latitude.setText("48.7")
            window._handle_metadata_line_edit_finished("latitude", window.metadata_latitude)
            window.metadata_notes.setPlainText("Проверка метаданных")
            window.property_file_name.setText("renamed")
            window._handle_file_name_edit_finished()
            window.save_project_file()

            loaded_project = load_project(project_path)
            record = loaded_project.images[0]

            self.assertEqual(record.display_name, "renamed.png")
            self.assertEqual(record.relative_path, "images/renamed.png")
            self.assertTrue((image_dir / "renamed.png").exists())
            self.assertFalse((image_dir / "leaf.png").exists())
            self.assertEqual(record.metadata["humidity"], "65")
            self.assertEqual(record.metadata["latitude"], "48.7")
            self.assertEqual(record.metadata["notes"], "Проверка метаданных")

    def test_file_name_button_renames_image_to_sample_id(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="rename_as_id",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                    )
                ],
            )
            project_path = root / "rename_as_id.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertEqual(window.rename_file_as_id_button.toolTip(), "Переименовать файл как ID")
            self.assertTrue(window.rename_file_as_id_button.isEnabled())

            window.metadata_sample_id.setText("SAMPLE_001")
            window._rename_file_to_sample_id()
            window.save_project_file()

            loaded_project = load_project(project_path)
            record = loaded_project.images[0]

            self.assertEqual(record.display_name, "SAMPLE_001.png")
            self.assertEqual(record.relative_path, "images/SAMPLE_001.png")
            self.assertEqual(record.metadata["sample_id"], "SAMPLE_001")
            self.assertTrue((image_dir / "SAMPLE_001.png").exists())
            self.assertFalse((image_dir / "leaf.png").exists())

    def test_sample_id_is_editable_and_unique(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf_a.png")
            Image.new("RGB", (20, 20), (10, 20, 30)).save(image_dir / "leaf_b.png")

            project = ProjectDocument(
                name="ids",
                images=[
                    ProjectImageRecord(
                        id="record-a",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                        metadata={"sample_id": "A"},
                    ),
                    ProjectImageRecord(
                        id="record-b",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                        metadata={"sample_id": "B"},
                    ),
                ],
            )
            project_path = root / "ids.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertEqual(window.metadata_sample_id.text(), "A")

            window.metadata_sample_id.setText("B")
            with patch.object(window, "_show_warning"):
                window._handle_metadata_line_edit_finished("sample_id", window.metadata_sample_id)
            self.assertEqual(window.metadata_sample_id.text(), "A")
            self.assertEqual(window.project_document.images[0].metadata["sample_id"], "A")

            window.metadata_sample_id.setText("C")
            window._handle_metadata_line_edit_finished("sample_id", window.metadata_sample_id)
            window.save_project_file()

            loaded_project = load_project(project_path)
            self.assertEqual(loaded_project.images[0].metadata["sample_id"], "C")

    def test_project_csv_export_contains_contour_statistics(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf_a.png")
            Image.new("RGB", (20, 20), (10, 20, 30)).save(image_dir / "leaf_b.png")

            annotation = Annotation(
                image_path="images/leaf_a.png",
                image_width=20,
                image_height=20,
                points=[
                    Point(1, 1),
                    Point(18, 1),
                    Point(18, 18),
                    Point(1, 18),
                ],
            )
            project = ProjectDocument(
                name="csv",
                images=[
                    ProjectImageRecord(
                        id="record-a",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                        annotation=annotation,
                        metadata={"sample_id": "A", "diagnosis": "class_a"},
                    ),
                    ProjectImageRecord(
                        id="record-b",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                        metadata={"sample_id": "B"},
                    ),
                    ProjectImageRecord(
                        id="record-missing",
                        relative_path="images/missing.png",
                        display_name="missing.png",
                        metadata={"sample_id": "M"},
                    ),
                ],
            )
            project_path = root / "csv.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))
            output_path = root / "report"

            with patch(
                "magicborder.main_window.QFileDialog.getSaveFileName",
                return_value=(str(output_path), ""),
            ):
                window.export_project_csv()

            csv_path = root / "report.csv"
            with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["id"], "A")
            self.assertEqual(rows[0]["file_name"], "leaf_a.png")
            self.assertEqual(rows[0]["diagnosis"], "class_a")
            self.assertEqual(rows[0]["r"], "120")
            self.assertEqual(rows[0]["g"], "80")
            self.assertEqual(rows[0]["b"], "40")
            self.assertEqual(rows[0]["contour_pixel_count"], "324")
            self.assertEqual(rows[1]["status"], "нет контура")
            self.assertEqual(rows[1]["r"], "")
            self.assertEqual(rows[2]["status"], "файл не найден")

    def test_datetime_metadata_fields_have_picker_buttons_and_validate_text(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="metadata",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                    )
                ],
            )
            project_path = root / "metadata.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertEqual(window.metadata_added_at_button.toolTip(), "Выбрать дату и время")
            self.assertTrue(_qdatetime_from_text("2026-05-03T10:20:30").isValid())

            window.metadata_captured_at.setText("2026-05-03 10:20:30")
            window._handle_metadata_line_edit_finished("captured_at", window.metadata_captured_at)
            self.assertEqual(
                window.project_document.images[0].metadata["captured_at"],
                "2026-05-03 10:20:30",
            )

            window.metadata_captured_at.setText("не дата")
            with patch.object(window, "_show_warning"):
                window._handle_metadata_line_edit_finished("captured_at", window.metadata_captured_at)
            self.assertEqual(window.metadata_captured_at.text(), "2026-05-03 10:20:30")

    def test_added_image_gets_added_at_metadata(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            source_dir.mkdir()
            source_path = source_dir / "leaf.png"
            Image.new("RGB", (20, 20), (120, 80, 40)).save(source_path)

            project_dir = root / "project"
            project_dir.mkdir()
            project_path = project_dir / "project.json"
            save_project(project_path, ProjectDocument(name="project", images=[]))

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            with patch(
                "magicborder.main_window.QFileDialog.getOpenFileNames",
                return_value=([str(source_path)], ""),
            ):
                window.add_images_to_project()

            loaded_project = load_project(project_path)
            record = loaded_project.images[0]

            self.assertTrue(record.metadata["added_at"])
            self.assertEqual(record.metadata["diagnosis"], "Не указано")
            self.assertTrue((project_dir / "images" / "leaf.png").exists())

    def test_invalid_metadata_value_is_not_saved(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="metadata",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                    )
                ],
            )
            project_path = root / "metadata.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            window.metadata_humidity.setText("150")
            with patch.object(window, "_show_warning"):
                window._handle_metadata_line_edit_finished("humidity", window.metadata_humidity)

            self.assertEqual(window.project_document.images[0].metadata["humidity"], "")
            self.assertEqual(window.metadata_humidity.text(), "")


if __name__ == "__main__":
    unittest.main()
