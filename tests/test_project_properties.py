from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree
from unittest.mock import patch
from uuid import UUID

from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPointF, Qt  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QAction,
    QApplication,
    QDialog,
    QSizePolicy,
    QHeaderView,
    QToolBar,
    QToolButton,
    QWidgetAction,
)

from magicborder.io_utils import load_project, save_project  # noqa: E402
from magicborder.main_window import (  # noqa: E402
    IMAGE_PROPERTY_EXPORT_KEYS,
    MainWindow,
    PROJECT_EXPORT_FIELDNAMES,
    _circle_contour_points,
    _qdatetime_from_text,
)
from magicborder.models import (  # noqa: E402
    Annotation,
    ImageCalibration,
    Point,
    ProjectDocument,
    ProjectImageRecord,
)
from magicborder.property_browser import PropertyBrowser, PropertyValueLabel  # noqa: E402

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _APP = app
    return app


def _assert_uuid4(test_case: unittest.TestCase, value: str) -> None:
    parsed_uuid = UUID(value)
    test_case.assertEqual(str(parsed_uuid), value)
    test_case.assertEqual(parsed_uuid.version, 4)


def _read_xlsx_rows(path: Path) -> list[list[str]]:
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as workbook:
        sheet_xml = workbook.read("xl/worksheets/sheet1.xml")

    root = ElementTree.fromstring(sheet_xml)
    values: list[list[str]] = []
    for row in root.findall("s:sheetData/s:row", namespace):
        row_values: dict[int, str] = {}
        for cell in row.findall("s:c", namespace):
            cell_ref = cell.attrib["r"]
            column_name = "".join(char for char in cell_ref if char.isalpha())
            text_node = cell.find("s:is/s:t", namespace)
            row_values[_xlsx_column_index(column_name)] = "" if text_node is None else (text_node.text or "")
        values.append([row_values.get(index, "") for index in range(max(row_values, default=-1) + 1)])
    return values


def _read_xlsx_dict_rows(path: Path) -> list[dict[str, str]]:
    values = _read_xlsx_rows(path)

    if not values:
        return []
    headers = values[0]
    return [
        dict(zip(headers, row + [""] * (len(headers) - len(row))))
        for row in values[1:]
    ]


def _xlsx_column_index(column_name: str) -> int:
    index = 0
    for char in column_name:
        index = index * 26 + ord(char.upper()) - 64
    return index - 1


def _minimal_export_window(root: Path) -> MainWindow:
    project = ProjectDocument(
        name="export",
        images=[
            ProjectImageRecord(
                id="row-1",
                relative_path="images/missing.png",
                display_name="missing.png",
                metadata={"diagnosis": "class_x"},
            )
        ],
    )
    project_path = root / "export.json"
    save_project(project_path, project)

    window = MainWindow()
    window._set_project(project_path, load_project(project_path))
    return window


def _list_item_color_name(window: MainWindow, row: int = 0) -> str:
    return window.project_list.item(row).foreground().color().name()


def _property_panel_rows(window: MainWindow) -> list[str]:
    return window.properties_browser.rows()


def _project_property_panel_rows(window: MainWindow) -> list[str]:
    return window.project_properties_browser.rows()


def _property_group(browser: PropertyBrowser, title: str):
    return browser.group_item(title)


def _main_toolbar(window: MainWindow) -> QToolBar:
    for toolbar in window.findChildren(QToolBar):
        if toolbar.windowTitle() == "Основная панель":
            return toolbar
    raise AssertionError("Основная панель не найдена")


class ProjectPropertiesTest(unittest.TestCase):
    def test_single_image_open_save_actions_are_removed(self) -> None:
        _app()
        window = MainWindow()

        self.assertFalse(hasattr(window, "open_image_action"))
        self.assertFalse(hasattr(window, "save_image_action"))

        actions = window.findChildren(QAction)
        action_texts = {action.text() for action in actions}
        shortcuts = {action.shortcut().toString() for action in actions if not action.shortcut().isEmpty()}

        self.assertNotIn("Открыть изображение...", action_texts)
        self.assertNotIn("Сохранить изображение...", action_texts)
        self.assertNotIn("Ctrl+O", shortcuts)
        self.assertNotIn("Ctrl+S", shortcuts)

    def test_exit_button_is_right_aligned_on_toolbar(self) -> None:
        _app()
        window = MainWindow()
        toolbar = _main_toolbar(window)

        actions = toolbar.actions()
        self.assertIs(actions[-1], window.exit_action)
        self.assertIsInstance(actions[-2], QWidgetAction)
        spacer = actions[-2].defaultWidget()
        self.assertIsNotNone(spacer)
        self.assertEqual(spacer.sizePolicy().horizontalPolicy(), QSizePolicy.Expanding)

    def test_image_properties_panel_uses_grouped_order(self) -> None:
        _app()
        window = MainWindow()

        self.assertEqual(
            _property_panel_rows(window),
            [
                "--- Общая информация о файле",
                "ID",
                "Файл",
                "Путь",
                "Размер",
                "Статус",
                "Дата добавления",
                "Дата съёмки",
                "--- Калибровка",
                "Длина калибровки, мм",
                "Масштаб",
                "--- Информация о главном контуре",
                "Аннотация",
                "Показывать контур",
                "Количество узлов контура",
                "Количество пикселов контура",
                "Площадь контура, мм²",
                "--- Измерения",
                "--- Углы",
                "Нет углов",
                "--- Отрезки",
                "Нет отрезков",
                "--- Цветовое пространство RGB",
                "Красный",
                "Зелёный",
                "Синий",
                "Средний цвет",
                "--- Цветовое пространство Lab",
                "L",
                "a",
                "b",
                "--- Цветовое пространство HSV",
                "H",
                "S",
                "V",
                "--- Цветовое пространство YUV",
                "Y",
                "U",
                "V",
                "--- Цветовое пространство LMS",
                "L",
                "M",
                "S",
                "--- Локация",
                "Освещённость",
                "Влажность, %",
                "Скорость ветра",
                "Направление ветра",
                "Широта",
                "Долгота",
                "--- Дополнительно",
                "Диагноз",
                "Дополнительные сведения",
            ],
        )
        calibration_group = _property_group(window.properties_browser, "Калибровка")
        contour_group = _property_group(window.properties_browser, "Информация о главном контуре")

        self.assertIs(
            window.properties_browser.property_item("Длина калибровки, мм").parent(),
            calibration_group,
        )
        self.assertIs(
            window.properties_browser.property_item("Масштаб").parent(),
            calibration_group,
        )
        self.assertIsNot(
            window.properties_browser.property_item("Длина калибровки, мм").parent(),
            contour_group,
        )
        self.assertIsNot(
            window.properties_browser.property_item("Масштаб").parent(),
            contour_group,
        )
        self.assertLess(
            window.properties_browser.indexOfTopLevelItem(calibration_group),
            window.properties_browser.indexOfTopLevelItem(contour_group),
        )

    def test_project_properties_panel_is_between_images_and_image_properties(self) -> None:
        _app()
        window = MainWindow()

        self.assertEqual(window.project_splitter.widget(0).objectName(), "projectImagesPanel")
        self.assertEqual(window.project_splitter.widget(1).objectName(), "projectPropertiesPanel")
        self.assertEqual(window.project_splitter.widget(2).objectName(), "imagePropertiesPanel")
        self.assertEqual(
            _project_property_panel_rows(window),
            [
                "--- Общие свойства",
                "Общая информация",
                "Количество изображений",
                "--- Цветовое пространство RGB",
                "Средний R",
                "Средний G",
                "Средний B",
                "--- Цветовое пространство Lab",
                "Средний L",
                "Средний a",
                "Средний b",
                "--- Цветовое пространство HSV",
                "Средний H",
                "Средний S",
                "Средний V",
                "--- Цветовое пространство YUV",
                "Средний Y",
                "Средний U",
                "Средний V",
                "--- Цветовое пространство LMS",
                "Средний L",
                "Средний M",
                "Средний S",
            ],
        )

    def test_left_project_sections_use_distinct_light_backgrounds(self) -> None:
        _app()
        window = MainWindow()
        style_sheet = window.project_splitter.parentWidget().styleSheet()

        self.assertIn("QWidget#projectImagesPanel { background: #f7fbff; }", style_sheet)
        self.assertIn("QWidget#projectPropertiesPanel { background: #f8fcf6; }", style_sheet)
        self.assertIn("QWidget#imagePropertiesPanel { background: #fffaf4; }", style_sheet)

    def test_property_groups_are_collapsed_by_default_and_toggle_independently(self) -> None:
        _app()
        window = MainWindow()

        browser = window.properties_browser
        self.assertIsInstance(browser, PropertyBrowser)
        file_group = _property_group(browser, "Общая информация о файле")
        contour_group = _property_group(browser, "Информация о главном контуре")

        self.assertFalse(file_group.isExpanded())
        self.assertFalse(browser.is_property_visible("ID"))
        self.assertFalse(browser.is_property_visible("Аннотация"))

        file_group.setExpanded(True)

        self.assertTrue(file_group.isExpanded())
        self.assertTrue(browser.is_property_visible("ID"))
        self.assertFalse(contour_group.isExpanded())
        self.assertFalse(browser.is_property_visible("Аннотация"))

        file_group.setExpanded(False)

        self.assertFalse(file_group.isExpanded())
        self.assertFalse(browser.is_property_visible("ID"))

    def test_project_property_groups_are_collapsed_by_default_and_toggle(self) -> None:
        _app()
        window = MainWindow()

        browser = window.project_properties_browser
        self.assertIsInstance(browser, PropertyBrowser)
        general_group = _property_group(browser, "Общие свойства")
        rgb_group = _property_group(browser, "Цветовое пространство RGB")

        self.assertFalse(general_group.isExpanded())
        self.assertFalse(browser.is_property_visible("Общая информация"))
        self.assertFalse(browser.is_property_visible("Средний R"))

        general_group.setExpanded(True)

        self.assertTrue(general_group.isExpanded())
        self.assertTrue(browser.is_property_visible("Общая информация"))
        self.assertFalse(rgb_group.isExpanded())
        self.assertFalse(browser.is_property_visible("Средний R"))

    def test_property_browsers_allow_key_column_width_adjustment(self) -> None:
        _app()
        window = MainWindow()

        for browser in (window.project_properties_browser, window.properties_browser):
            browser.resize(320, 220)

            self.assertIsInstance(browser, PropertyBrowser)
            self.assertFalse(browser.isHeaderHidden())
            self.assertEqual(browser.header().sectionResizeMode(0), QHeaderView.Interactive)
            self.assertEqual(browser.header().sectionResizeMode(1), QHeaderView.Stretch)

            browser.set_key_column_width(110)
            self.assertEqual(browser.key_column_width(), 110)
            browser.set_key_column_width(180)
            self.assertEqual(browser.key_column_width(), 180)

        self.assertIsInstance(window.project_image_count, PropertyValueLabel)
        self.assertIsInstance(window.property_path, PropertyValueLabel)

    def test_property_browser_long_values_remain_available_and_resize_rows(self) -> None:
        _app()
        browser = PropertyBrowser()
        browser.resize(260, 220)
        browser.add_group("Группа", expanded=True)
        label = PropertyValueLabel()
        item = browser.add_property("Группа", "Длинное свойство", label)

        browser.set_key_column_width(96)
        compact_height = item.sizeHint(1).height()
        long_text = "длинное значение свойства " * 20
        label.setText(long_text)
        browser.refresh_layout()

        self.assertTrue(label.wordWrap())
        self.assertTrue(label.textInteractionFlags() & Qt.TextSelectableByMouse)
        self.assertEqual(label.toolTip(), long_text)
        self.assertGreater(item.sizeHint(1).height(), compact_height)

    def test_image_properties_panel_has_no_refresh_button(self) -> None:
        _app()
        window = MainWindow()

        property_buttons = window.image_properties_panel.findChildren(QToolButton)
        button_texts = {button.text() for button in property_buttons}
        button_tooltips = {button.toolTip() for button in property_buttons}

        self.assertNotIn("Обновить", button_texts)
        self.assertNotIn("Обновить свойства выбранного изображения", button_tooltips)

    def test_project_panel_header_buttons_share_icon_text_style(self) -> None:
        _app()
        window = MainWindow()

        buttons = [
            window.export_project_excel_button,
            window.export_image_properties_excel_button,
        ]

        for button in buttons:
            self.assertEqual(button.text(), "Excel")
            self.assertEqual(button.toolButtonStyle(), Qt.ToolButtonTextBesideIcon)
            self.assertFalse(button.icon().isNull())
            self.assertFalse(button.autoRaise())

    def test_new_project_prompts_for_name_then_parent_directory(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            call_order: list[str] = []

            def get_project_name(*_args, **_kwargs) -> tuple[str, bool]:
                call_order.append("name")
                return ("Leaves 2026", True)

            def get_parent_directory(*_args, **_kwargs) -> str:
                call_order.append("directory")
                return str(root)

            window = MainWindow()
            with patch("magicborder.main_window.QInputDialog.getText", side_effect=get_project_name), patch(
                "magicborder.main_window.QFileDialog.getExistingDirectory",
                side_effect=get_parent_directory,
            ):
                window.new_project()

            project_dir = root / "Leaves_2026"
            project_path = project_dir / "Leaves_2026.json"

            self.assertEqual(call_order, ["name", "directory"])
            self.assertEqual(window.project_path, project_path.resolve())
            self.assertTrue(project_dir.is_dir())
            self.assertTrue((project_dir / "images").is_dir())
            self.assertTrue(project_path.is_file())
            self.assertEqual(load_project(project_path).name, "Leaves_2026")

    def test_new_project_cancel_name_does_not_open_directory_dialog(self) -> None:
        _app()
        window = MainWindow()

        with patch("magicborder.main_window.QInputDialog.getText", return_value=("", False)), patch(
            "magicborder.main_window.QFileDialog.getExistingDirectory",
        ) as get_directory:
            window.new_project()

        get_directory.assert_not_called()
        self.assertIsNone(window.project_path)
        self.assertIsNone(window.project_document)

    def test_new_project_cancel_directory_creates_nothing(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = MainWindow()

            with patch("magicborder.main_window.QInputDialog.getText", return_value=("Leaves 2026", True)), patch(
                "magicborder.main_window.QFileDialog.getExistingDirectory",
                return_value="",
            ):
                window.new_project()

            self.assertFalse((root / "Leaves_2026").exists())
            self.assertIsNone(window.project_path)
            self.assertIsNone(window.project_document)

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

            self.assertTrue(window.canvas.has_image())
            self.assertEqual(window.canvas.current_image_path(), image_dir / "leaf.png")
            self.assertEqual(window.property_points.text(), "5")
            self.assertEqual(window.property_contour_area_mm2.text(), "калибровка не произведена")
            self.assertEqual(window.property_calibration_length.text(), "калибровка не произведена")
            self.assertFalse(window.property_calibration_length.isEnabled())
            self.assertEqual(window.property_calibration_scale.text(), "-")

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
            self.assertEqual(window.property_contour_area_mm2.text(), "калибровка не произведена")
            self.assertEqual(window.property_calibration_length.text(), "калибровка не произведена")
            self.assertEqual(len(window.project_document.images[0].annotation.points), 4)

            window.delete_current_contour()

            self.assertFalse(window.canvas.has_contour())
            self.assertIsNone(window.project_document.images[0].annotation)
            self.assertEqual(window.property_annotation.text(), "нет")
            self.assertEqual(window.property_points.text(), "-")
            self.assertEqual(window.property_contour_pixels.text(), "-")
            self.assertEqual(window.property_contour_area_mm2.text(), "-")
            self.assertEqual(window.property_calibration_length.text(), "калибровка не произведена")
            self.assertEqual(window.property_calibration_scale.text(), "-")
            self.assertEqual(window.property_lab_l.text(), "-")
            self.assertEqual(window.property_lab_a.text(), "-")
            self.assertEqual(window.property_lab_b.text(), "-")
            self.assertEqual(window.property_hsv_h.text(), "-")
            self.assertEqual(window.property_hsv_s.text(), "-")
            self.assertEqual(window.property_hsv_v.text(), "-")
            self.assertEqual(window.property_yuv_y.text(), "-")
            self.assertEqual(window.property_yuv_u.text(), "-")
            self.assertEqual(window.property_yuv_v.text(), "-")
            self.assertEqual(window.property_lms_l.text(), "-")
            self.assertEqual(window.property_lms_m.text(), "-")
            self.assertEqual(window.property_lms_s.text(), "-")

    def test_contour_visibility_property_toggles_canvas_without_changing_annotation(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "empty.png")
            Image.new("RGB", (20, 20), (60, 120, 80)).save(image_dir / "leaf.png")

            annotation = Annotation(
                image_path="images/leaf.png",
                image_width=20,
                image_height=20,
                points=[
                    Point(2, 2),
                    Point(17, 2),
                    Point(17, 17),
                    Point(2, 17),
                ],
            )
            project = ProjectDocument(
                name="contour_visibility",
                images=[
                    ProjectImageRecord(
                        id="empty",
                        relative_path="images/empty.png",
                        display_name="empty.png",
                        image_width=20,
                        image_height=20,
                    ),
                    ProjectImageRecord(
                        id="leaf",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=20,
                        image_height=20,
                        annotation=annotation,
                    ),
                ],
            )
            project_path = root / "contour_visibility.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertIn("Показывать контур", _property_panel_rows(window))
            self.assertFalse(window.property_contour_visible.isEnabled())
            self.assertFalse(window.property_contour_visible.isChecked())

            window.project_list.setCurrentRow(1)

            record = window._project_image_by_id("leaf")
            self.assertIsNotNone(record)
            self.assertTrue(window.property_contour_visible.isEnabled())
            self.assertTrue(window.property_contour_visible.isChecked())
            self.assertTrue(window.canvas._path_item.isVisible())
            self.assertTrue(all(handle.isVisible() for handle in window.canvas._handles))

            window._project_autosave_timer.stop()
            window.property_contour_visible.setChecked(False)

            self.assertFalse(window.canvas.is_contour_visible())
            self.assertFalse(window.canvas._path_item.isVisible())
            self.assertTrue(all(not handle.isVisible() for handle in window.canvas._handles))
            self.assertIsNotNone(record.annotation)
            self.assertEqual(record.annotation.points, annotation.points)
            self.assertFalse(window._project_autosave_timer.isActive())

            window.property_contour_visible.setChecked(True)

            self.assertTrue(window.canvas.is_contour_visible())
            self.assertTrue(window.canvas._path_item.isVisible())
            self.assertTrue(all(handle.isVisible() for handle in window.canvas._handles))

    def test_angle_measurements_are_project_entities_and_saved(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf_a.png")
            Image.new("RGB", (20, 20), (10, 20, 30)).save(image_dir / "leaf_b.png")

            project = ProjectDocument(
                name="measurements",
                images=[
                    ProjectImageRecord(
                        id="leaf-a",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                        image_width=20,
                        image_height=20,
                    ),
                    ProjectImageRecord(
                        id="leaf-b",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                        image_width=20,
                        image_height=20,
                    ),
                ],
            )
            project_path = root / "angles.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertTrue(window.measure_angle_action.isEnabled())
            self.assertFalse(window.delete_angle_action.isEnabled())

            window.canvas.set_angle_measurements(
                [
                    (
                        Point(2, 10),
                        Point(2, 2),
                        Point(10, 2),
                    )
                ]
            )

            self.assertTrue(window._project_autosave_timer.isActive())
            record = window._project_image_by_id("leaf-a")
            self.assertIsNotNone(record)
            self.assertEqual(len(record.measurements.angles), 1)
            angle_id = record.measurements.angles[0].id
            self.assertIn("--- Измерения", _property_panel_rows(window))
            self.assertIn("--- Углы", _property_panel_rows(window))
            self.assertIn("--- Угол 1", _property_panel_rows(window))
            self.assertIn("Имя", _property_panel_rows(window))
            self.assertIn("Показывать на канвасе", _property_panel_rows(window))
            angle_id_label = window.properties_browser.itemWidget(
                window.properties_browser.property_item(f"angle:{angle_id}:id"),
                1,
            )
            self.assertEqual(angle_id_label.text(), angle_id)
            visibility_field = window._angle_visibility_fields[angle_id]
            self.assertTrue(visibility_field.isEnabled())
            self.assertTrue(visibility_field.isChecked())
            self.assertTrue(window.canvas.is_angle_measurement_visible(angle_id))

            name_field = window._angle_name_fields[angle_id]
            self.assertEqual(name_field.text(), "")
            name_field.setText("Контрольный угол")
            window._handle_angle_name_edit_finished(angle_id, name_field)
            self.assertEqual(record.measurements.angles[0].name, "Контрольный угол")
            self.assertIn("--- Контрольный угол", _property_panel_rows(window))
            self.assertEqual(
                window.canvas._angle_graphics[0].label.toPlainText(),
                "Контрольный угол: 90°",
            )

            note_field = window._angle_note_fields[angle_id]
            note_field.setText("контрольный угол")
            window._handle_angle_note_edit_finished(angle_id, note_field)
            self.assertEqual(record.measurements.angles[0].note, "контрольный угол")

            window.canvas._angle_graphics[0].handles[1].setSelected(True)
            self.assertTrue(window.delete_angle_action.isEnabled())

            visibility_field = window._angle_visibility_fields[angle_id]
            window._project_autosave_timer.stop()
            visibility_field.setChecked(False)

            self.assertFalse(visibility_field.isChecked())
            self.assertFalse(window.canvas.is_angle_measurement_visible(angle_id))
            self.assertFalse(window.canvas._angle_graphics[0].label.isVisible())
            self.assertFalse(window.canvas.has_selected_angle_vertex())
            self.assertFalse(window.delete_angle_action.isEnabled())
            self.assertEqual(len(record.measurements.angles), 1)
            self.assertFalse(window._project_autosave_timer.isActive())

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            saved_angle = payload["images"][0]["measurements"]["angles"][0]
            self.assertNotIn("visible", saved_angle)

            visibility_field.setChecked(True)

            self.assertTrue(window.canvas.is_angle_measurement_visible(angle_id))
            self.assertTrue(window.canvas._angle_graphics[0].label.isVisible())

            window.canvas._angle_graphics[0].handles[1].setSelected(True)

            self.assertTrue(window.delete_angle_action.isEnabled())

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            saved_angle = payload["images"][0]["measurements"]["angles"][0]
            self.assertEqual(saved_angle["id"], angle_id)
            self.assertEqual(saved_angle["first"], {"x": 2.0, "y": 10.0})
            self.assertEqual(saved_angle["vertex"], {"x": 2.0, "y": 2.0})
            self.assertEqual(saved_angle["second"], {"x": 10.0, "y": 2.0})
            self.assertEqual(saved_angle["name"], "Контрольный угол")
            self.assertEqual(saved_angle["note"], "контрольный угол")
            self.assertNotIn("visible", saved_angle)

            window.project_list.setCurrentRow(1)

            self.assertEqual(window.canvas.angle_measurements(), [])

            window.project_list.setCurrentRow(0)

            self.assertEqual(len(window.canvas.angle_measurements()), 1)
            self.assertEqual(window.canvas.angle_measurement_records()[0][0], angle_id)
            self.assertEqual(
                window.canvas._angle_graphics[0].label.toPlainText(),
                "Контрольный угол: 90°",
            )

            window.canvas._angle_graphics[0].handles[1].setSelected(True)
            window.delete_selected_angle()

            self.assertEqual(window.canvas.angle_measurements(), [])
            self.assertEqual(record.measurements.angles, [])
            self.assertFalse(window.delete_angle_action.isEnabled())

    def test_segment_measurements_are_project_entities_and_saved(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (40, 30), (120, 80, 40)).save(image_dir / "leaf_a.png")
            Image.new("RGB", (40, 30), (10, 20, 30)).save(image_dir / "leaf_b.png")

            project = ProjectDocument(
                name="segments",
                images=[
                    ProjectImageRecord(
                        id="leaf-a",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                        image_width=40,
                        image_height=30,
                        calibration=ImageCalibration(
                            start=Point(0, 0),
                            end=Point(10, 0),
                            length_mm=5,
                        ),
                    ),
                    ProjectImageRecord(
                        id="leaf-b",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                        image_width=40,
                        image_height=30,
                    ),
                ],
            )
            project_path = root / "segments.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertTrue(window.measure_segment_action.isEnabled())
            self.assertFalse(window.delete_segment_action.isEnabled())
            self.assertIn("--- Отрезки", _property_panel_rows(window))
            self.assertIn("Нет отрезков", _property_panel_rows(window))

            window.canvas.set_segment_measurement_records(
                [("segment-a", Point(2, 2), Point(12, 2), "A", "B")]
            )

            record = window._project_image_by_id("leaf-a")
            self.assertIsNotNone(record)
            self.assertEqual(len(record.measurements.segments), 1)
            segment_id = record.measurements.segments[0].id
            self.assertEqual(segment_id, "segment-a")
            self.assertIn("--- Отрезок 1", _property_panel_rows(window))
            self.assertIn("Имя", _property_panel_rows(window))
            self.assertIn("Показывать на канвасе", _property_panel_rows(window))
            self.assertIn("Подпись первой точки", _property_panel_rows(window))
            self.assertIn("Подпись второй точки", _property_panel_rows(window))
            self.assertIn("Примечание", _property_panel_rows(window))
            segment_length_label = window.properties_browser.itemWidget(
                window.properties_browser.property_item(f"segment:{segment_id}:length"),
                1,
            )
            self.assertEqual(segment_length_label.text(), "10 px / 5 мм")
            self.assertEqual(
                window.canvas._segment_graphics[0].length_label.toPlainText(),
                "Отрезок 1\n10 px / 5 мм",
            )
            visibility_field = window._segment_visibility_fields[segment_id]
            self.assertTrue(visibility_field.isEnabled())
            self.assertTrue(visibility_field.isChecked())
            self.assertTrue(window.canvas.is_segment_measurement_visible(segment_id))

            name_field = window._segment_name_fields[segment_id]
            self.assertEqual(name_field.text(), "")
            name_field.setText("Контрольный отрезок")
            window._handle_segment_name_edit_finished(segment_id, name_field)
            self.assertEqual(record.measurements.segments[0].name, "Контрольный отрезок")
            self.assertIn("--- Контрольный отрезок", _property_panel_rows(window))
            self.assertEqual(
                window.canvas._segment_graphics[0].length_label.toPlainText(),
                "Контрольный отрезок\n10 px / 5 мм",
            )

            start_label_field = window._segment_start_label_fields[segment_id]
            start_label_field.setText("Начало")
            window._handle_segment_label_edit_finished(segment_id, start_label_field, "start")
            end_label_field = window._segment_end_label_fields[segment_id]
            end_label_field.setText("Конец")
            window._handle_segment_label_edit_finished(segment_id, end_label_field, "end")
            note_field = window._segment_note_fields[segment_id]
            note_field.setText("измерить повторно")
            window._handle_segment_note_edit_finished(segment_id, note_field)

            self.assertEqual(record.measurements.segments[0].name, "Контрольный отрезок")
            self.assertEqual(record.measurements.segments[0].start_label, "Начало")
            self.assertEqual(record.measurements.segments[0].end_label, "Конец")
            self.assertEqual(record.measurements.segments[0].note, "измерить повторно")
            self.assertEqual(window.canvas._segment_graphics[0].start_label.toPlainText(), "Начало")
            self.assertEqual(window.canvas._segment_graphics[0].end_label.toPlainText(), "Конец")

            window.canvas._segment_graphics[0].handles[0].setSelected(True)

            self.assertTrue(window.delete_segment_action.isEnabled())

            visibility_field = window._segment_visibility_fields[segment_id]
            window._project_autosave_timer.stop()
            visibility_field.setChecked(False)

            self.assertFalse(visibility_field.isChecked())
            self.assertFalse(window.canvas.is_segment_measurement_visible(segment_id))
            self.assertFalse(window.canvas._segment_graphics[0].line.isVisible())
            self.assertFalse(window.canvas._segment_graphics[0].length_label.isVisible())
            self.assertFalse(window.canvas.has_selected_segment_endpoint())
            self.assertFalse(window.delete_segment_action.isEnabled())
            self.assertEqual(len(record.measurements.segments), 1)
            self.assertFalse(window._project_autosave_timer.isActive())

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            saved_segment = payload["images"][0]["measurements"]["segments"][0]
            self.assertNotIn("visible", saved_segment)

            visibility_field.setChecked(True)

            self.assertTrue(window.canvas.is_segment_measurement_visible(segment_id))
            self.assertTrue(window.canvas._segment_graphics[0].line.isVisible())
            self.assertTrue(window.canvas._segment_graphics[0].length_label.isVisible())

            window.canvas._segment_graphics[0].handles[0].setSelected(True)
            self.assertTrue(window.delete_segment_action.isEnabled())

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            saved_segment = payload["images"][0]["measurements"]["segments"][0]
            self.assertEqual(saved_segment["id"], segment_id)
            self.assertEqual(saved_segment["name"], "Контрольный отрезок")
            self.assertEqual(saved_segment["start"], {"x": 2.0, "y": 2.0})
            self.assertEqual(saved_segment["end"], {"x": 12.0, "y": 2.0})
            self.assertEqual(saved_segment["start_label"], "Начало")
            self.assertEqual(saved_segment["end_label"], "Конец")
            self.assertEqual(saved_segment["note"], "измерить повторно")
            self.assertNotIn("visible", saved_segment)

            window.project_list.setCurrentRow(1)

            self.assertEqual(window.canvas.segment_measurements(), [])

            window.project_list.setCurrentRow(0)

            self.assertEqual(len(window.canvas.segment_measurements()), 1)
            self.assertEqual(window.canvas.segment_measurement_records()[0][0], segment_id)
            self.assertEqual(
                window.canvas._segment_graphics[0].length_label.toPlainText(),
                "Контрольный отрезок\n10 px / 5 мм",
            )
            self.assertEqual(window._segment_name_fields[segment_id].text(), "Контрольный отрезок")
            self.assertEqual(window._segment_note_fields[segment_id].text(), "измерить повторно")

            window.canvas._segment_graphics[0].handles[0].setSelected(True)
            window.delete_selected_segment()

            self.assertEqual(window.canvas.segment_measurements(), [])
            self.assertEqual(record.measurements.segments, [])
            self.assertFalse(window.delete_segment_action.isEnabled())

    def test_image_calibration_scale_round_trip_edit_reset_and_export(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (140, 30), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="calibration",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=140,
                        image_height=30,
                        calibration=ImageCalibration(
                            start=Point(1, 2),
                            end=Point(121, 2),
                            length_mm=10,
                        ),
                    )
                ],
            )
            project_path = root / "calibration.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertTrue(window.canvas.has_calibration())
            self.assertTrue(window.calibrate_scale_action.isEnabled())
            self.assertTrue(window.reset_calibration_action.isEnabled())
            self.assertEqual(
                window.property_calibration_scale.text(),
                "120 px = 10 мм; 12 px/мм; 0.0833 мм/px",
            )

            window.canvas.calibration_handle_moved(1, QPointF(61, 2))

            record = window.project_document.images[0]
            self.assertIsNotNone(record.calibration)
            self.assertEqual(record.calibration.end, Point(61, 2))
            self.assertEqual(
                window.property_calibration_scale.text(),
                "60 px = 10 мм; 6 px/мм; 0.1667 мм/px",
            )

            export_path = root / "calibration_properties.xlsx"
            window._write_image_properties_excel(export_path, ["calibration_scale"])
            self.assertEqual(
                _read_xlsx_dict_rows(export_path),
                [
                    {
                        "Свойство": "Масштаб",
                        "Значение": "60 px = 10 мм; 6 px/мм; 0.1667 мм/px",
                    },
                ],
            )

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["images"][0]["calibration"],
                {
                    "start": {"x": 1.0, "y": 2.0},
                    "end": {"x": 61.0, "y": 2.0},
                    "length_mm": 10.0,
                },
            )

            window.reset_current_calibration()

            self.assertFalse(window.canvas.has_calibration())
            self.assertIsNone(record.calibration)
            self.assertFalse(window.reset_calibration_action.isEnabled())
            self.assertEqual(window.property_calibration_scale.text(), "-")
            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertNotIn("calibration", payload["images"][0])

            with patch("magicborder.main_window.QInputDialog.getDouble", return_value=(5.0, False)):
                window._handle_calibration_segment_selected([Point(2, 2), Point(22, 2)])

            self.assertFalse(window.canvas.has_calibration())
            self.assertIsNone(record.calibration)

            with patch("magicborder.main_window.QInputDialog.getDouble", return_value=(5.0, True)):
                window._handle_calibration_segment_selected([Point(2, 2), Point(22, 2)])

            self.assertTrue(window.canvas.has_calibration())
            self.assertIsNotNone(record.calibration)
            self.assertEqual(record.calibration.length_mm, 5.0)
            self.assertEqual(
                window.property_calibration_scale.text(),
                "20 px = 5 мм; 4 px/мм; 0.25 мм/px",
            )

    def test_image_contour_area_mm2_updates_with_contour_and_calibration(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (30, 30), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="area_mm2",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=30,
                        image_height=30,
                        annotation=Annotation(
                            image_path="images/leaf.png",
                            image_width=30,
                            image_height=30,
                            points=[
                                Point(1, 1),
                                Point(18, 1),
                                Point(18, 18),
                                Point(1, 18),
                            ],
                        ),
                        calibration=ImageCalibration(
                            start=Point(1, 1),
                            end=Point(19, 1),
                            length_mm=6,
                        ),
                    )
                ],
            )
            project_path = root / "area_mm2.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertEqual(window.property_contour_pixels.text(), "324")
            self.assertEqual(window.property_contour_area_mm2.text(), "36 мм²")
            self.assertEqual(window.property_calibration_length.text(), "6 мм")
            self.assertTrue(window.property_calibration_length.isEnabled())
            self.assertEqual(window.canvas.calibration_label_text(), "6 мм")
            self.assertTrue(window.canvas._calibration_label_item.isVisible())

            window.canvas.set_contour(
                [
                    Point(1, 1),
                    Point(9, 1),
                    Point(9, 9),
                    Point(1, 9),
                ]
            )

            self.assertEqual(window.property_contour_pixels.text(), "81")
            self.assertEqual(window.property_contour_area_mm2.text(), "9 мм²")

            window.property_calibration_length.setText("3,0")
            window._handle_calibration_length_edit_finished()

            record = window.project_document.images[0]
            self.assertIsNotNone(record.calibration)
            self.assertEqual(record.calibration.start, Point(1, 1))
            self.assertEqual(record.calibration.end, Point(19, 1))
            self.assertEqual(record.calibration.length_mm, 3.0)
            self.assertEqual(window.property_calibration_length.text(), "3 мм")
            self.assertEqual(
                window.property_calibration_scale.text(),
                "18 px = 3 мм; 6 px/мм; 0.1667 мм/px",
            )
            self.assertEqual(window.property_contour_area_mm2.text(), "2.25 мм²")
            self.assertEqual(window.canvas.calibration_label_text(), "3 мм")

            with patch.object(window, "_show_warning") as show_warning:
                window.property_calibration_length.setText("0")
                window._handle_calibration_length_edit_finished()

            show_warning.assert_called_once_with(
                "Некорректная калибровка",
                "Длина калибровочного отрезка должна быть положительным числом.",
            )
            self.assertEqual(record.calibration.length_mm, 3.0)
            self.assertEqual(window.property_calibration_length.text(), "3 мм")

            label_position = window.canvas._calibration_label_item.pos()

            window.canvas.calibration_handle_moved(1, QPointF(10, 1))

            self.assertNotEqual(window.canvas._calibration_label_item.pos(), label_position)
            self.assertEqual(window.canvas.calibration_label_text(), "3 мм")
            self.assertEqual(window.property_contour_area_mm2.text(), "9 мм²")

            export_path = root / "area_properties.xlsx"
            window._write_image_properties_excel(
                export_path,
                ["calibration_length_mm", "contour_area_mm2"],
            )
            self.assertEqual(
                _read_xlsx_dict_rows(export_path),
                [
                    {
                        "Свойство": "Длина калибровки, мм",
                        "Значение": "3 мм",
                    },
                    {
                        "Свойство": "Площадь контура, мм²",
                        "Значение": "9 мм²",
                    },
                ],
            )

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertNotIn("contour_area_mm2", payload["images"][0])
            self.assertEqual(payload["images"][0]["calibration"]["length_mm"], 3.0)

            window.reset_current_calibration()

            self.assertEqual(
                window.property_contour_area_mm2.text(),
                "калибровка не произведена",
            )
            self.assertEqual(
                window.property_calibration_length.text(),
                "калибровка не произведена",
            )
            self.assertFalse(window.property_calibration_length.isEnabled())
            self.assertFalse(window.canvas._calibration_label_item.isVisible())

    def test_image_properties_lab_values_update_export_and_stay_out_of_json(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            image = Image.new("RGB", (20, 20), (10, 20, 30))
            for x in range(10):
                for y in range(20):
                    image.putpixel((x, y), (120, 80, 40))
            image.save(image_dir / "leaf.png")

            left_contour = [
                Point(1, 1),
                Point(8, 1),
                Point(8, 18),
                Point(1, 18),
            ]
            project = ProjectDocument(
                name="lab_values",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                        image_width=20,
                        image_height=20,
                        annotation=Annotation(
                            image_path="images/leaf.png",
                            image_width=20,
                            image_height=20,
                            points=left_contour,
                        ),
                    )
                ],
            )
            project_path = root / "lab_values.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertEqual(window.property_lab_l.text(), "38")
            self.assertEqual(window.property_lab_a.text(), "12")
            self.assertEqual(window.property_lab_b.text(), "30")
            self.assertEqual(window.property_hsv_h.text(), "30")
            self.assertEqual(window.property_hsv_s.text(), "170")
            self.assertEqual(window.property_hsv_v.text(), "120")
            self.assertEqual(window.property_yuv_y.text(), "87")
            self.assertEqual(window.property_yuv_u.text(), "105")
            self.assertEqual(window.property_yuv_v.text(), "157")
            self.assertEqual(window.property_lms_l.text(), "255")
            self.assertEqual(window.property_lms_m.text(), "255")
            self.assertEqual(window.property_lms_s.text(), "255")

            window.canvas.set_contour(
                [
                    Point(11, 1),
                    Point(18, 1),
                    Point(18, 18),
                    Point(11, 18),
                ]
            )

            self.assertEqual(window.property_lab_l.text(), "6")
            self.assertEqual(window.property_lab_a.text(), "0")
            self.assertEqual(window.property_lab_b.text(), "-8")
            self.assertEqual(window.property_hsv_h.text(), "210")
            self.assertEqual(window.property_hsv_s.text(), "170")
            self.assertEqual(window.property_hsv_v.text(), "30")
            self.assertEqual(window.property_yuv_y.text(), "18")
            self.assertEqual(window.property_yuv_u.text(), "134")
            self.assertEqual(window.property_yuv_v.text(), "121")
            self.assertEqual(window.property_lms_l.text(), "255")
            self.assertEqual(window.property_lms_m.text(), "255")
            self.assertEqual(window.property_lms_s.text(), "255")

            export_path = root / "lab_properties.xlsx"
            window._write_image_properties_excel(
                export_path,
                [
                    "lab_l",
                    "lab_a",
                    "lab_b",
                    "hsv_h",
                    "hsv_s",
                    "hsv_v",
                    "yuv_y",
                    "yuv_u",
                    "yuv_v",
                    "lms_l",
                    "lms_m",
                    "lms_s",
                ],
            )
            self.assertEqual(
                _read_xlsx_dict_rows(export_path),
                [
                    {"Свойство": "L", "Значение": "6"},
                    {"Свойство": "a", "Значение": "0"},
                    {"Свойство": "b", "Значение": "-8"},
                    {"Свойство": "H", "Значение": "210"},
                    {"Свойство": "S", "Значение": "170"},
                    {"Свойство": "V", "Значение": "30"},
                    {"Свойство": "Y", "Значение": "18"},
                    {"Свойство": "U", "Значение": "134"},
                    {"Свойство": "V", "Значение": "121"},
                    {"Свойство": "L", "Значение": "255"},
                    {"Свойство": "M", "Значение": "255"},
                    {"Свойство": "S", "Значение": "255"},
                ],
            )

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertNotIn("lab_l", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("lab_a", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("lab_b", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("hsv_h", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("yuv_y", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("lms_l", json.dumps(payload, ensure_ascii=False))

    def test_project_properties_save_general_info_and_count_images(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = ProjectDocument(
                name="project_info",
                images=[
                    ProjectImageRecord(
                        id="leaf-1",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                    ),
                    ProjectImageRecord(
                        id="leaf-2",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                    ),
                ],
            )
            project_path = root / "project_info.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertEqual(window.project_image_count.text(), "2")
            window.project_general_info.setPlainText("Серия измерений 2026")
            window.save_project_file()

            loaded_project = load_project(project_path)
            self.assertEqual(loaded_project.project_info.general_info, "Серия измерений 2026")

            reopened = MainWindow()
            reopened._set_project(project_path, load_project(project_path))
            self.assertEqual(reopened.project_general_info.toPlainText(), "Серия измерений 2026")
            self.assertEqual(reopened.project_image_count.text(), "2")
            self.assertEqual(reopened.project_mean_lab_l.text(), "-")
            self.assertEqual(reopened.project_mean_lab_a.text(), "-")
            self.assertEqual(reopened.project_mean_lab_b.text(), "-")
            self.assertEqual(reopened.project_mean_hsv_h.text(), "-")
            self.assertEqual(reopened.project_mean_hsv_s.text(), "-")
            self.assertEqual(reopened.project_mean_hsv_v.text(), "-")
            self.assertEqual(reopened.project_mean_yuv_y.text(), "-")
            self.assertEqual(reopened.project_mean_yuv_u.text(), "-")
            self.assertEqual(reopened.project_mean_yuv_v.text(), "-")
            self.assertEqual(reopened.project_mean_lms_l.text(), "-")
            self.assertEqual(reopened.project_mean_lms_m.text(), "-")
            self.assertEqual(reopened.project_mean_lms_s.text(), "-")

    def test_project_properties_average_rgb_uses_all_available_contours(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (100, 50, 0)).save(image_dir / "leaf_a.png")
            Image.new("RGB", (20, 20), (10, 20, 30)).save(image_dir / "leaf_b.png")
            Image.new("RGB", (20, 20), (200, 200, 200)).save(image_dir / "no_contour.png")

            contour = [
                Point(1, 1),
                Point(18, 1),
                Point(18, 18),
                Point(1, 18),
            ]
            project = ProjectDocument(
                name="project_rgb",
                images=[
                    ProjectImageRecord(
                        id="leaf-a",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                        image_width=20,
                        image_height=20,
                        annotation=Annotation(
                            image_path="images/leaf_a.png",
                            image_width=20,
                            image_height=20,
                            points=contour,
                        ),
                    ),
                    ProjectImageRecord(
                        id="leaf-b",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                        image_width=20,
                        image_height=20,
                        annotation=Annotation(
                            image_path="images/leaf_b.png",
                            image_width=20,
                            image_height=20,
                            points=contour,
                        ),
                    ),
                    ProjectImageRecord(
                        id="no-contour",
                        relative_path="images/no_contour.png",
                        display_name="no_contour.png",
                    ),
                    ProjectImageRecord(
                        id="missing",
                        relative_path="images/missing.png",
                        display_name="missing.png",
                        annotation=Annotation(
                            image_path="images/missing.png",
                            image_width=20,
                            image_height=20,
                            points=contour,
                        ),
                    ),
                ],
            )
            project_path = root / "project_rgb.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertEqual(window.project_mean_red.text(), "55")
            self.assertEqual(window.project_mean_green.text(), "35")
            self.assertEqual(window.project_mean_blue.text(), "15")
            self.assertEqual(window.project_mean_lab_l.text(), "16")
            self.assertEqual(window.project_mean_lab_a.text(), "10")
            self.assertEqual(window.project_mean_lab_b.text(), "14")
            self.assertEqual(window.project_mean_hsv_h.text(), "120")
            self.assertEqual(window.project_mean_hsv_s.text(), "212")
            self.assertEqual(window.project_mean_hsv_v.text(), "65")
            self.assertEqual(window.project_mean_yuv_y.text(), "38")
            self.assertEqual(window.project_mean_yuv_u.text(), "116")
            self.assertEqual(window.project_mean_yuv_v.text(), "142")
            self.assertEqual(window.project_mean_lms_l.text(), "137")
            self.assertEqual(window.project_mean_lms_m.text(), "152")
            self.assertEqual(window.project_mean_lms_s.text(), "180")

            window.canvas.set_contour(
                [
                    Point(1, 1),
                    Point(4, 1),
                    Point(4, 4),
                    Point(1, 4),
                ]
            )

            self.assertEqual(window.project_mean_red.text(), "14")
            self.assertEqual(window.project_mean_green.text(), "21")
            self.assertEqual(window.project_mean_blue.text(), "29")
            self.assertEqual(window.project_mean_lab_l.text(), "7")
            self.assertEqual(window.project_mean_lab_a.text(), "1")
            self.assertEqual(window.project_mean_lab_b.text(), "-6")
            self.assertEqual(window.project_mean_hsv_h.text(), "202")
            self.assertEqual(window.project_mean_hsv_s.text(), "174")
            self.assertEqual(window.project_mean_hsv_v.text(), "33")
            self.assertEqual(window.project_mean_yuv_y.text(), "20")
            self.assertEqual(window.project_mean_yuv_u.text(), "132")
            self.assertEqual(window.project_mean_yuv_v.text(), "123")
            self.assertEqual(window.project_mean_lms_l.text(), "30")
            self.assertEqual(window.project_mean_lms_m.text(), "58")
            self.assertEqual(window.project_mean_lms_s.text(), "248")

            window.save_project_file()
            payload = json.loads(project_path.read_text(encoding="utf-8"))
            self.assertNotIn("project_mean_lab_l", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("project_mean_lab_a", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("project_mean_lab_b", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("project_mean_hsv_h", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("project_mean_yuv_y", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("project_mean_lms_l", json.dumps(payload, ensure_ascii=False))

    def test_project_list_item_color_tracks_annotation_state(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project = ProjectDocument(
                name="list_colors",
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
            project_path = root / "list_colors.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertIn("без аннотации", window.project_list.item(0).text())
            self.assertEqual(_list_item_color_name(window), "#b42318")

            window.canvas.set_contour(
                [
                    Point(1, 1),
                    Point(18, 1),
                    Point(18, 18),
                    Point(1, 18),
                ]
            )

            self.assertIn("аннотация есть", window.project_list.item(0).text())
            self.assertEqual(_list_item_color_name(window), "#1f2937")

            window.delete_current_contour()

            self.assertIn("без аннотации", window.project_list.item(0).text())
            self.assertEqual(_list_item_color_name(window), "#b42318")

    def test_project_list_item_color_marks_annotation_errors_red(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")

            project_path = root / "list_error.json"
            project_path.write_text(
                json.dumps(
                    {
                        "name": "list_error",
                        "images": [
                            {
                                "file": {
                                    "id": "leaf-1",
                                    "path": "images/leaf.png",
                                    "display_name": "leaf.png",
                                    "image_size": {"width": 20, "height": 20},
                                },
                                "contour": {
                                    "annotation": {
                                        "image_path": "images/leaf.png",
                                        "image_size": {"width": 20, "height": 20},
                                        "points": [],
                                    },
                                },
                                "location": {},
                                "details": {},
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertIn("ошибка аннотации", window.project_list.item(0).text())
            self.assertEqual(_list_item_color_name(window), "#b42318")

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

            payload = json.loads(project_path.read_text(encoding="utf-8"))
            loaded_project = load_project(project_path)
            record = loaded_project.images[0]
            record_payload = payload["images"][0]

            self.assertNotIn("metadata", record_payload)
            self.assertEqual(record_payload["file"]["display_name"], "renamed.png")
            self.assertEqual(record_payload["file"]["path"], "images/renamed.png")
            self.assertEqual(record_payload["location"]["humidity"], "65")
            self.assertEqual(record_payload["location"]["latitude"], "48.7")
            self.assertEqual(record_payload["details"]["notes"], "Проверка метаданных")
            self.assertEqual(record.display_name, "renamed.png")
            self.assertEqual(record.relative_path, "images/renamed.png")
            self.assertTrue((image_dir / "renamed.png").exists())
            self.assertFalse((image_dir / "leaf.png").exists())
            self.assertEqual(record.metadata["humidity"], "65")
            self.assertEqual(record.metadata["latitude"], "48.7")
            self.assertEqual(record.metadata["notes"], "Проверка метаданных")

    def test_file_name_button_renames_image_to_record_id(self) -> None:
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

            window.image_id.setText("SAMPLE_001")
            window._rename_file_to_image_id()
            window.save_project_file()

            loaded_project = load_project(project_path)
            record = loaded_project.images[0]

            self.assertEqual(record.display_name, "SAMPLE_001.png")
            self.assertEqual(record.relative_path, "images/SAMPLE_001.png")
            self.assertEqual(record.id, "SAMPLE_001")
            self.assertNotIn("sample_id", record.metadata)
            self.assertTrue((image_dir / "SAMPLE_001.png").exists())
            self.assertFalse((image_dir / "leaf.png").exists())

    def test_generate_image_id_button_creates_new_uuid(self) -> None:
        _app()
        window = MainWindow()

        self.assertFalse(window.generate_image_id_button.isEnabled())
        self.assertEqual(window.generate_image_id_button.toolTip(), "Сгенерировать ID")
        self.assertEqual(
            window.generate_image_id_button.statusTip(),
            "Сгенерировать новый GUID изображения.",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "leaf.png")
            Image.new("RGB", (20, 20), (10, 20, 30)).save(image_dir / "other.png")

            project = ProjectDocument(
                name="generate_id",
                images=[
                    ProjectImageRecord(
                        id="11111111-1111-4111-8111-111111111111",
                        relative_path="images/leaf.png",
                        display_name="leaf.png",
                    ),
                    ProjectImageRecord(
                        id="22222222-2222-4222-8222-222222222222",
                        relative_path="images/other.png",
                        display_name="other.png",
                    ),
                ],
            )
            project_path = root / "generate_id.json"
            save_project(project_path, project)

            window._set_project(project_path, load_project(project_path))

            self.assertTrue(window.generate_image_id_button.isEnabled())
            old_id = window.project_document.images[0].id

            window._generate_image_id()
            first_generated_id = window.project_document.images[0].id
            window._generate_image_id()
            second_generated_id = window.project_document.images[0].id
            window.save_project_file()

            loaded_project = load_project(project_path)
            self.assertNotEqual(first_generated_id, old_id)
            self.assertNotEqual(second_generated_id, first_generated_id)
            _assert_uuid4(self, first_generated_id)
            _assert_uuid4(self, second_generated_id)
            self.assertEqual(window.image_id.text(), second_generated_id)
            self.assertEqual(loaded_project.images[0].id, second_generated_id)
            self.assertEqual(loaded_project.images[1].id, "22222222-2222-4222-8222-222222222222")
            self.assertNotIn("sample_id", loaded_project.images[0].metadata)

            output_path = root / "generated_ids"
            with patch.object(window, "_select_project_export_columns", return_value=PROJECT_EXPORT_FIELDNAMES), patch(
                "magicborder.main_window.QFileDialog.getSaveFileName",
                return_value=(str(output_path), ""),
            ):
                window.export_project_excel()

            xlsx_path = root / "generated_ids.xlsx"
            rows = _read_xlsx_dict_rows(xlsx_path)
            self.assertEqual(rows[0]["ID изображения"], second_generated_id)

    def test_record_id_is_editable_and_unique(self) -> None:
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
                        id="A",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                    ),
                    ProjectImageRecord(
                        id="B",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                    ),
                ],
            )
            project_path = root / "ids.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertEqual(window.image_id.text(), "A")

            window.image_id.setText("B")
            with patch.object(window, "_show_warning"):
                window._handle_image_id_edit_finished()
            self.assertEqual(window.image_id.text(), "A")
            self.assertEqual(window.project_document.images[0].id, "A")

            window.image_id.setText("C")
            window._handle_image_id_edit_finished()
            window.save_project_file()

            loaded_project = load_project(project_path)
            self.assertEqual(loaded_project.images[0].id, "C")
            self.assertEqual(window.project_list.currentItem().data(Qt.UserRole), "C")

    def test_project_excel_export_contains_contour_statistics(self) -> None:
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
                name="excel",
                images=[
                    ProjectImageRecord(
                        id="record-a",
                        relative_path="images/leaf_a.png",
                        display_name="leaf_a.png",
                        annotation=annotation,
                        metadata={"diagnosis": "class_a"},
                    ),
                    ProjectImageRecord(
                        id="record-b",
                        relative_path="images/leaf_b.png",
                        display_name="leaf_b.png",
                    ),
                    ProjectImageRecord(
                        id="record-missing",
                        relative_path="images/missing.png",
                        display_name="missing.png",
                    ),
                ],
            )
            project_path = root / "excel.json"
            save_project(project_path, project)

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))
            output_path = root / "report"

            self.assertEqual(window.export_project_excel_action.text(), "Экспорт списка в Excel...")
            self.assertEqual(window.export_project_excel_action.toolTip(), "Экспорт списка в Excel")
            self.assertEqual(window.export_project_excel_button.toolTip(), "Экспорт списка в Excel")

            with patch.object(window, "_select_project_export_columns", return_value=PROJECT_EXPORT_FIELDNAMES), patch(
                "magicborder.main_window.QFileDialog.getSaveFileName",
                return_value=(str(output_path), ""),
            ) as get_save_file_name:
                window.export_project_excel()

            self.assertEqual(get_save_file_name.call_args.args[1], "Экспорт списка в Excel")
            xlsx_path = root / "report.xlsx"
            raw_rows = _read_xlsx_rows(xlsx_path)
            rows = _read_xlsx_dict_rows(xlsx_path)

            self.assertEqual(
                raw_rows[0],
                [
                    "ID изображения",
                    "Имя файла",
                    "Относительный путь",
                    "Есть аннотация",
                    "Статус",
                    "Диагноз",
                    "Средний R",
                    "Средний G",
                    "Средний B",
                    "Количество пикселов контура",
                ],
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["ID изображения"], "record-a")
            self.assertEqual(rows[0]["Имя файла"], "leaf_a.png")
            self.assertEqual(rows[0]["Диагноз"], "class_a")
            self.assertEqual(rows[0]["Средний R"], "120")
            self.assertEqual(rows[0]["Средний G"], "80")
            self.assertEqual(rows[0]["Средний B"], "40")
            self.assertEqual(rows[0]["Количество пикселов контура"], "324")
            self.assertEqual(rows[1]["Статус"], "нет контура")
            self.assertEqual(rows[1]["Средний R"], "")
            self.assertEqual(rows[2]["Статус"], "файл не найден")

    def test_project_excel_export_uses_selected_columns_only(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)
            output_path = root / "selected"

            with patch.object(window, "_select_project_export_columns", return_value=["file_name", "diagnosis"]), patch(
                "magicborder.main_window.QFileDialog.getSaveFileName",
                return_value=(str(output_path), ""),
            ):
                window.export_project_excel()

            xlsx_path = root / "selected.xlsx"
            raw_rows = _read_xlsx_rows(xlsx_path)
            rows = _read_xlsx_dict_rows(xlsx_path)

            self.assertEqual(raw_rows[0], ["Имя файла", "Диагноз"])
            self.assertEqual(rows, [{"Имя файла": "missing.png", "Диагноз": "class_x"}])

    def test_project_excel_export_cancel_column_selection_does_not_open_file_dialog(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)

            with patch.object(window, "_select_project_export_columns", return_value=None), patch(
                "magicborder.main_window.QFileDialog.getSaveFileName",
            ) as get_save_file_name:
                window.export_project_excel()

            get_save_file_name.assert_not_called()

    def test_project_excel_export_requires_at_least_one_column(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)

            with patch.object(window, "_select_project_export_columns", return_value=[]), patch.object(
                window,
                "_show_warning",
            ) as show_warning, patch(
                "magicborder.main_window.QFileDialog.getSaveFileName",
            ) as get_save_file_name:
                window.export_project_excel()

            show_warning.assert_called_once_with(
                "Нет выбранных столбцов",
                "Выберите хотя бы один столбец для экспорта.",
            )
            get_save_file_name.assert_not_called()

    def test_project_excel_column_dialog_selects_all_columns_by_default(self) -> None:
        _app()
        window = MainWindow()

        with patch("magicborder.main_window.QDialog.exec_", return_value=QDialog.Accepted):
            selected_fieldnames = window._select_project_export_columns()

        self.assertEqual(selected_fieldnames, PROJECT_EXPORT_FIELDNAMES)

    def test_image_properties_excel_export_uses_selected_properties_vertically(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)
            output_path = root / "properties"

            self.assertTrue(window.export_image_properties_excel_action.isEnabled())
            self.assertTrue(window.export_image_properties_excel_button.isEnabled())
            self.assertEqual(
                window.export_image_properties_excel_action.text(),
                "Экспорт свойств изображения в Excel...",
            )
            self.assertEqual(
                window.export_image_properties_excel_button.toolTip(),
                "Экспорт свойств изображения в Excel",
            )

            with patch.object(
                window,
                "_select_image_property_export_items",
                return_value=["file_name", "diagnosis", "status"],
            ), patch(
                "magicborder.main_window.QFileDialog.getSaveFileName",
                return_value=(str(output_path), ""),
            ) as get_save_file_name:
                window.export_image_properties_excel()

            self.assertEqual(get_save_file_name.call_args.args[1], "Экспорт свойств изображения в Excel")
            xlsx_path = root / "properties.xlsx"
            raw_rows = _read_xlsx_rows(xlsx_path)
            rows = _read_xlsx_dict_rows(xlsx_path)

            self.assertEqual(raw_rows[0], ["Свойство", "Значение"])
            self.assertEqual(
                rows,
                [
                    {"Свойство": "Файл", "Значение": "missing.png"},
                    {"Свойство": "Диагноз", "Значение": "class_x"},
                    {"Свойство": "Статус", "Значение": "отсутствует"},
                ],
            )

    def test_image_properties_excel_export_cancel_property_selection_does_not_open_file_dialog(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)

            with patch.object(window, "_select_image_property_export_items", return_value=None), patch(
                "magicborder.main_window.QFileDialog.getSaveFileName",
            ) as get_save_file_name:
                window.export_image_properties_excel()

            get_save_file_name.assert_not_called()

    def test_image_properties_excel_export_requires_at_least_one_property(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            window = _minimal_export_window(root)

            with patch.object(window, "_select_image_property_export_items", return_value=[]), patch.object(
                window,
                "_show_warning",
            ) as show_warning, patch(
                "magicborder.main_window.QFileDialog.getSaveFileName",
            ) as get_save_file_name:
                window.export_image_properties_excel()

            show_warning.assert_called_once_with(
                "Нет выбранных свойств",
                "Выберите хотя бы одно свойство для экспорта.",
            )
            get_save_file_name.assert_not_called()

    def test_image_properties_excel_export_requires_selected_image(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_path = root / "empty.json"
            save_project(project_path, ProjectDocument(name="empty", images=[]))

            window = MainWindow()
            window._set_project(project_path, load_project(project_path))

            self.assertFalse(window.export_image_properties_excel_action.isEnabled())
            self.assertFalse(window.export_image_properties_excel_button.isEnabled())

            with patch.object(window, "_show_warning") as show_warning, patch(
                "magicborder.main_window.QFileDialog.getSaveFileName",
            ) as get_save_file_name:
                window.export_image_properties_excel()

            show_warning.assert_called_once_with(
                "Нет выбранного изображения",
                "Выберите изображение в списке проекта.",
            )
            get_save_file_name.assert_not_called()

    def test_image_properties_export_dialog_selects_all_properties_by_default(self) -> None:
        _app()
        window = MainWindow()

        with patch("magicborder.main_window.QDialog.exec_", return_value=QDialog.Accepted):
            selected_properties = window._select_image_property_export_items()

        self.assertEqual(selected_properties, IMAGE_PROPERTY_EXPORT_KEYS)

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

            _assert_uuid4(self, record.id)
            self.assertTrue(record.metadata["added_at"])
            self.assertEqual(record.metadata["diagnosis"], "Не указано")
            self.assertNotIn("sample_id", record.metadata)
            self.assertTrue((project_dir / "images" / "leaf.png").exists())

    def test_sync_project_images_folder_adds_untracked_images_only(self) -> None:
        _app()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_dir = root / "project"
            image_dir = project_dir / "images"
            nested_dir = image_dir / "nested"
            nested_dir.mkdir(parents=True)
            Image.new("RGB", (20, 20), (120, 80, 40)).save(image_dir / "existing.png")
            Image.new("RGB", (18, 12), (10, 20, 30)).save(image_dir / "new.png")
            Image.new("RGB", (16, 14), (40, 50, 60)).save(nested_dir / "nested.jpg")
            (image_dir / "broken.png").write_bytes(b"not an image")
            (image_dir / "ignored.txt").write_text("not an image", encoding="utf-8")

            project_path = project_dir / "project.json"
            project = ProjectDocument(
                name="project",
                images=[
                    ProjectImageRecord(
                        id="existing-id",
                        relative_path="images/existing.png",
                        display_name="existing.png",
                    )
                ],
            )
            save_project(project_path, project)

            window = MainWindow()

            self.assertFalse(window.sync_images_action.isEnabled())
            self.assertEqual(window.sync_images_action.toolTip(), "Синхронизировать папку изображений")
            self.assertEqual(
                window.sync_images_action.statusTip(),
                "Добавить в проект изображения из папки проекта, которых ещё нет в JSON.",
            )

            window._set_project(project_path, load_project(project_path))

            self.assertTrue(window.sync_images_action.isEnabled())

            with patch.object(window, "_show_warning") as show_warning:
                window.sync_project_images_folder()
            self.assertTrue(show_warning.called)

            loaded_project = load_project(project_path)
            records_by_path = {record.relative_path: record for record in loaded_project.images}

            self.assertEqual(len(loaded_project.images), 3)
            self.assertIn("images/existing.png", records_by_path)
            self.assertIn("images/new.png", records_by_path)
            self.assertIn("images/nested/nested.jpg", records_by_path)
            self.assertNotIn("images/broken.png", records_by_path)
            self.assertNotIn("images/ignored.txt", records_by_path)

            new_record = records_by_path["images/new.png"]
            nested_record = records_by_path["images/nested/nested.jpg"]
            self.assertIsNone(new_record.annotation)
            self.assertIsNone(nested_record.annotation)
            self.assertEqual(new_record.display_name, "new.png")
            self.assertEqual((new_record.image_width, new_record.image_height), (18, 12))
            self.assertEqual(nested_record.display_name, "nested.jpg")
            self.assertEqual((nested_record.image_width, nested_record.image_height), (16, 14))
            self.assertTrue(new_record.metadata["added_at"])
            self.assertEqual(new_record.metadata["diagnosis"], "Не указано")
            self.assertNotIn("sample_id", new_record.metadata)
            _assert_uuid4(self, new_record.id)
            _assert_uuid4(self, nested_record.id)

            with patch.object(window, "_show_warning"):
                window.sync_project_images_folder()
            reloaded_project = load_project(project_path)
            self.assertEqual(len(reloaded_project.images), 3)

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
