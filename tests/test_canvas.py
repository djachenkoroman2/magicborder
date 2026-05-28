from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QMouseEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication, QGraphicsView  # noqa: E402

from magicborder.canvas import ImageCanvas  # noqa: E402
from magicborder.io_utils import loaded_image_from_rgb_array  # noqa: E402
from magicborder.models import Point  # noqa: E402

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    _APP = app
    return app


def _canvas_with_image() -> ImageCanvas:
    _app()
    canvas = ImageCanvas()
    canvas.resize(320, 240)
    rgb_array = np.zeros((80, 100, 3), dtype=np.uint8)
    canvas.set_loaded_image(loaded_image_from_rgb_array(Path("leaf.png"), rgb_array))
    return canvas


def _mouse_press(
    canvas: ImageCanvas,
    scene_point: QPointF,
    button: Qt.MouseButton = Qt.LeftButton,
) -> None:
    view_point = canvas.mapFromScene(scene_point)
    event = QMouseEvent(
        QEvent.MouseButtonPress,
        view_point,
        button,
        button,
        Qt.NoModifier,
    )
    canvas.mousePressEvent(event)


def _mouse_move(canvas: ImageCanvas, scene_point: QPointF) -> None:
    view_point = canvas.mapFromScene(scene_point)
    event = QMouseEvent(
        QEvent.MouseMove,
        view_point,
        Qt.NoButton,
        Qt.NoButton,
        Qt.NoModifier,
    )
    canvas.mouseMoveEvent(event)


def _calibration_label_center(canvas: ImageCanvas) -> QPointF:
    item = canvas._calibration_label_item
    rect = item.boundingRect()
    scale_x = abs(canvas.transform().m11()) or 1.0
    scale_y = abs(canvas.transform().m22()) or scale_x
    return item.pos() + QPointF(rect.center().x() / scale_x, rect.center().y() / scale_y)


def _path_signature(path) -> list[tuple[float, float]]:
    return [
        (round(path.elementAt(index).x, 3), round(path.elementAt(index).y, 3))
        for index in range(path.elementCount())
    ]


class ImageCanvasCalibrationPreviewTest(unittest.TestCase):
    def test_calibration_preview_marks_first_point_updates_and_emits_segment(self) -> None:
        canvas = _canvas_with_image()
        selected_segments: list[list[Point]] = []
        canvas.calibration_segment_selected.connect(selected_segments.append)

        canvas.begin_calibration()
        self.assertEqual(canvas.dragMode(), QGraphicsView.NoDrag)

        _mouse_press(canvas, QPointF(10, 12))

        self.assertFalse(canvas.has_calibration())
        self.assertEqual(len(canvas._calibration_capture_points), 1)
        self.assertTrue(canvas._calibration_preview_start_item.isVisible())
        self.assertTrue(canvas._calibration_preview_line_item.isVisible())

        start_marker = canvas._calibration_preview_start_item.pos()
        self.assertAlmostEqual(start_marker.x(), 10.0, delta=0.5)
        self.assertAlmostEqual(start_marker.y(), 12.0, delta=0.5)

        _mouse_move(canvas, QPointF(75, 36))

        preview_line = canvas._calibration_preview_line_item.line()
        self.assertAlmostEqual(preview_line.x1(), 10.0, delta=0.5)
        self.assertAlmostEqual(preview_line.y1(), 12.0, delta=0.5)
        self.assertAlmostEqual(preview_line.x2(), 75.0, delta=0.5)
        self.assertAlmostEqual(preview_line.y2(), 36.0, delta=0.5)

        _mouse_press(canvas, QPointF(90, 42))

        self.assertEqual(len(selected_segments), 1)
        self.assertFalse(canvas._calibration_capture_active)
        self.assertEqual(canvas._calibration_capture_points, [])
        self.assertFalse(canvas._calibration_preview_start_item.isVisible())
        self.assertFalse(canvas._calibration_preview_line_item.isVisible())
        self.assertEqual(canvas.dragMode(), QGraphicsView.ScrollHandDrag)

        start, end = selected_segments[0]
        self.assertAlmostEqual(start.x, 10.0, delta=0.5)
        self.assertAlmostEqual(start.y, 12.0, delta=0.5)
        self.assertAlmostEqual(end.x, 90.0, delta=0.5)
        self.assertAlmostEqual(end.y, 42.0, delta=0.5)

    def test_calibration_preview_cancel_preserves_existing_calibration(self) -> None:
        canvas = _canvas_with_image()
        canvas.set_calibration(Point(1, 2), Point(31, 2))

        canvas.begin_calibration()
        _mouse_press(canvas, QPointF(10, 12))
        _mouse_move(canvas, QPointF(75, 36))

        self.assertTrue(canvas._calibration_preview_line_item.isVisible())

        canvas.cancel_calibration()

        self.assertTrue(canvas.has_calibration())
        self.assertEqual(canvas.calibration_points(), [Point(1, 2), Point(31, 2)])
        self.assertFalse(canvas._calibration_preview_start_item.isVisible())
        self.assertFalse(canvas._calibration_preview_line_item.isVisible())
        self.assertEqual(canvas.dragMode(), QGraphicsView.ScrollHandDrag)

    def test_calibration_label_updates_moves_and_clears_with_segment(self) -> None:
        canvas = _canvas_with_image()

        canvas.set_calibration(Point(10, 20), Point(70, 20), "10 мм")

        self.assertTrue(canvas._calibration_label_item.isVisible())
        self.assertEqual(canvas.calibration_label_text(), "10 мм")
        self.assertEqual(canvas._calibration_label_item.toPlainText(), "10 мм")
        label_center = _calibration_label_center(canvas)
        self.assertAlmostEqual(label_center.x(), 40.0, delta=0.5)
        self.assertAlmostEqual(label_center.y(), 20.0, delta=0.5)
        initial_position = canvas._calibration_label_item.pos()

        canvas.zoom_in()

        label_center = _calibration_label_center(canvas)
        self.assertAlmostEqual(label_center.x(), 40.0, delta=0.5)
        self.assertAlmostEqual(label_center.y(), 20.0, delta=0.5)

        canvas.set_calibration_label_text("12.5 мм")

        self.assertTrue(canvas._calibration_label_item.isVisible())
        self.assertEqual(canvas.calibration_label_text(), "12.5 мм")
        self.assertEqual(canvas._calibration_label_item.toPlainText(), "12.5 мм")

        canvas.calibration_handle_moved(1, QPointF(70, 45))

        self.assertNotEqual(canvas._calibration_label_item.pos(), initial_position)
        label_center = _calibration_label_center(canvas)
        self.assertAlmostEqual(label_center.x(), 40.0, delta=0.5)
        self.assertAlmostEqual(label_center.y(), 32.5, delta=0.5)

        canvas.clear_calibration()

        self.assertFalse(canvas._calibration_label_item.isVisible())
        self.assertEqual(canvas.calibration_label_text(), "")


class ImageCanvasAngleMeasurementTest(unittest.TestCase):
    def test_angle_tool_creates_preview_and_measurement_from_three_clicks(self) -> None:
        canvas = _canvas_with_image()

        canvas.begin_angle_measurement()
        self.assertEqual(canvas.dragMode(), QGraphicsView.NoDrag)

        _mouse_press(canvas, QPointF(10, 20))

        self.assertEqual(len(canvas._angle_capture_points), 1)
        self.assertTrue(canvas._angle_preview_first_item.isVisible())
        self.assertTrue(canvas._angle_preview_line_item.isVisible())

        _mouse_move(canvas, QPointF(30, 40))

        preview_line = canvas._angle_preview_line_item.line()
        self.assertAlmostEqual(preview_line.x1(), 10.0, delta=0.5)
        self.assertAlmostEqual(preview_line.y1(), 20.0, delta=0.5)
        self.assertAlmostEqual(preview_line.x2(), 30.0, delta=0.5)
        self.assertAlmostEqual(preview_line.y2(), 40.0, delta=0.5)

        _mouse_press(canvas, QPointF(10, 10))

        self.assertEqual(len(canvas._angle_capture_points), 2)
        self.assertTrue(canvas._angle_preview_vertex_item.isVisible())
        self.assertTrue(canvas._angle_preview_fixed_line_item.isVisible())

        _mouse_move(canvas, QPointF(20, 10))

        second_preview_line = canvas._angle_preview_line_item.line()
        self.assertAlmostEqual(second_preview_line.x1(), 10.0, delta=0.5)
        self.assertAlmostEqual(second_preview_line.y1(), 10.0, delta=0.5)
        self.assertAlmostEqual(second_preview_line.x2(), 20.0, delta=0.5)
        self.assertAlmostEqual(second_preview_line.y2(), 10.0, delta=0.5)

        _mouse_press(canvas, QPointF(20, 10))

        self.assertTrue(canvas._angle_capture_active)
        self.assertEqual(canvas._angle_capture_points, [])
        self.assertFalse(canvas._angle_preview_first_item.isVisible())
        self.assertFalse(canvas._angle_preview_vertex_item.isVisible())
        self.assertFalse(canvas._angle_preview_line_item.isVisible())
        self.assertEqual(len(canvas.angle_measurements()), 1)
        self.assertEqual(canvas._angle_graphics[0].label.toPlainText(), "Угол 1: 90°")
        self.assertTrue(canvas._angle_graphics[0].arc.isVisible())
        self.assertFalse(canvas._angle_graphics[0].arc.path().isEmpty())
        self.assertNotEqual(
            canvas._angle_graphics[0].arc.pen().color().name(),
            canvas._angle_graphics[0].first_line.pen().color().name(),
        )

        first, vertex, second = canvas.angle_measurements()[0]
        self.assertAlmostEqual(first.x, 10.0, delta=0.5)
        self.assertAlmostEqual(first.y, 20.0, delta=0.5)
        self.assertAlmostEqual(vertex.x, 10.0, delta=0.5)
        self.assertAlmostEqual(vertex.y, 10.0, delta=0.5)
        self.assertAlmostEqual(second.x, 20.0, delta=0.5)
        self.assertAlmostEqual(second.y, 10.0, delta=0.5)

    def test_angle_handles_recalculate_and_delete_by_selected_vertex(self) -> None:
        canvas = _canvas_with_image()
        canvas.set_angle_measurements(
            [
                (
                    Point(10, 20),
                    Point(10, 10),
                    Point(20, 10),
                )
            ]
        )

        self.assertTrue(canvas.has_angle_measurements())
        self.assertEqual(canvas._angle_graphics[0].label.toPlainText(), "Угол 1: 90°")
        initial_arc = _path_signature(canvas._angle_graphics[0].arc.path())
        self.assertTrue(canvas._angle_graphics[0].arc.isVisible())

        canvas.angle_handle_moved(0, 2, QPointF(20, 20))

        self.assertEqual(canvas.angle_measurements()[0][2], Point(20, 20))
        self.assertEqual(canvas._angle_graphics[0].label.toPlainText(), "Угол 1: 45°")
        self.assertNotEqual(initial_arc, _path_signature(canvas._angle_graphics[0].arc.path()))

        canvas._angle_graphics[0].handles[0].setSelected(True)
        self.assertFalse(canvas.has_selected_angle_vertex())
        self.assertFalse(canvas.delete_selected_angle())
        self.assertTrue(canvas.has_angle_measurements())

        canvas._angle_graphics[0].handles[0].setSelected(False)
        canvas._angle_graphics[0].handles[1].setSelected(True)

        self.assertTrue(canvas.has_selected_angle_vertex())
        self.assertTrue(canvas.delete_selected_angle())
        self.assertEqual(canvas.angle_measurements(), [])
        self.assertFalse(canvas.has_angle_measurements())

    def test_angle_labels_follow_collection_order_and_renumber_after_delete(self) -> None:
        canvas = _canvas_with_image()
        canvas.set_angle_measurement_records(
            [
                ("angle-a", Point(10, 20), Point(10, 10), Point(20, 10)),
                ("angle-b", Point(30, 30), Point(30, 10), Point(50, 30)),
                ("angle-c", Point(60, 30), Point(60, 10), Point(80, 10)),
            ]
        )

        self.assertEqual(
            [graphics.label.toPlainText() for graphics in canvas._angle_graphics],
            ["Угол 1: 90°", "Угол 2: 45°", "Угол 3: 90°"],
        )

        canvas._angle_graphics[1].handles[1].setSelected(True)

        self.assertTrue(canvas.delete_selected_angle())
        self.assertEqual(
            [graphics.label.toPlainText() for graphics in canvas._angle_graphics],
            ["Угол 1: 90°", "Угол 2: 90°"],
        )
        self.assertEqual(
            [record[0] for record in canvas.angle_measurement_records()],
            ["angle-a", "angle-c"],
        )

    def test_angle_capture_cancel_clears_preview_without_creating_measurement(self) -> None:
        canvas = _canvas_with_image()

        canvas.begin_angle_measurement()
        _mouse_press(canvas, QPointF(10, 20))
        _mouse_move(canvas, QPointF(30, 40))

        self.assertTrue(canvas._angle_preview_line_item.isVisible())

        _mouse_press(canvas, QPointF(30, 40), Qt.RightButton)

        self.assertFalse(canvas._angle_capture_active)
        self.assertEqual(canvas._angle_capture_points, [])
        self.assertFalse(canvas._angle_preview_first_item.isVisible())
        self.assertFalse(canvas._angle_preview_line_item.isVisible())
        self.assertEqual(canvas.angle_measurements(), [])
        self.assertEqual(canvas.dragMode(), QGraphicsView.ScrollHandDrag)


if __name__ == "__main__":
    unittest.main()
