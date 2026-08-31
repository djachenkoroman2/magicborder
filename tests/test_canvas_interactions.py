from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PyQt5.QtWidgets import QGraphicsView

from magicborder.canvas import (
    ImageCanvas,
    NodeHandleItem,
    _angle_arc_path,
    _calibration_length_mm_from_text,
    _compact_float,
    _distance_to_segment,
    _format_angle_degrees,
)
from magicborder.io_utils import loaded_image_from_rgb_array
from magicborder.models import ANGLE_LINE_COLOR, SEGMENT_LABEL_COLOR, Point

SQUARE = [Point(10, 10), Point(90, 10), Point(90, 70), Point(10, 70)]
TRIANGLE = [Point(10, 10), Point(90, 10), Point(50, 70)]


@pytest.fixture()
def view(canvas_with_image: ImageCanvas) -> ImageCanvas:
    """Канвас 100x80 с масштабом 1:1 для предсказуемых допусков попадания."""
    canvas_with_image.resetTransform()
    return canvas_with_image


@pytest.fixture()
def messages(view: ImageCanvas) -> list[str]:
    collected: list[str] = []
    view.message_changed.connect(collected.append)
    return collected


def _key_press(canvas: ImageCanvas, key: int, modifiers=Qt.NoModifier) -> QKeyEvent:
    event = QKeyEvent(QEvent.KeyPress, key, modifiers)
    canvas.keyPressEvent(event)
    return event


def _mouse_press(
    canvas: ImageCanvas,
    scene_point: QPointF,
    button: Qt.MouseButton = Qt.LeftButton,
) -> QMouseEvent:
    event = QMouseEvent(
        QEvent.MouseButtonPress,
        canvas.mapFromScene(scene_point),
        button,
        button,
        Qt.NoModifier,
    )
    canvas.mousePressEvent(event)
    return event


def _double_click(canvas: ImageCanvas, scene_point: QPointF) -> QMouseEvent:
    event = QMouseEvent(
        QEvent.MouseButtonDblClick,
        canvas.mapFromScene(scene_point),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    canvas.mouseDoubleClickEvent(event)
    return event


def _wheel(
    canvas: ImageCanvas, delta_y: int, modifiers=Qt.ControlModifier
) -> QWheelEvent:
    position = QPointF(canvas.viewport().rect().center())
    event = QWheelEvent(
        position,
        position,
        QPoint(0, 0),
        QPoint(0, delta_y),
        Qt.NoButton,
        modifiers,
        Qt.NoScrollPhase,
        False,
    )
    canvas.wheelEvent(event)
    return event


def _scale(canvas: ImageCanvas) -> float:
    return canvas.transform().m11()


class TestKeyPressEscape:
    def test_escape_cancels_angle_capture(
        self, view: ImageCanvas, messages: list[str]
    ) -> None:
        view.begin_angle_measurement()

        event = _key_press(view, Qt.Key_Escape)

        assert view._angle_capture_active is False
        assert event.isAccepted() is True
        assert "Измерение угла отменено." in messages

    def test_escape_cancels_segment_capture(
        self, view: ImageCanvas, messages: list[str]
    ) -> None:
        view.begin_segment_measurement()

        event = _key_press(view, Qt.Key_Escape)

        assert view._segment_capture_active is False
        assert event.isAccepted() is True
        assert "Измерение отрезка отменено." in messages

    def test_escape_cancels_calibration_capture(
        self,
        view: ImageCanvas,
        messages: list[str],
    ) -> None:
        view.begin_calibration()

        event = _key_press(view, Qt.Key_Escape)

        assert view._calibration_capture_active is False
        assert event.isAccepted() is True
        assert "Калибровка отменена." in messages

    def test_escape_without_capture_is_passed_through(self, view: ImageCanvas) -> None:
        event = _key_press(view, Qt.Key_Escape)

        assert event.isAccepted() is False


class TestKeyPressDeletePriority:
    def _prepare(self, view: ImageCanvas) -> None:
        view.set_contour(SQUARE)
        view.set_angle_measurements([[Point(20, 20), Point(20, 40), Point(40, 40)]])
        view.set_segment_measurements([[Point(60, 20), Point(80, 40)]])

    def test_angle_vertex_is_deleted_first(self, view: ImageCanvas) -> None:
        self._prepare(view)
        view._angle_graphics[0].handles[1].setSelected(True)
        view._segment_graphics[0].handles[0].setSelected(True)
        view._handles[0].setSelected(True)

        event = _key_press(view, Qt.Key_Delete)

        assert event.isAccepted() is True
        assert view.has_angle_measurements() is False
        assert view.has_segment_measurements() is True
        assert len(view.contour_points()) == 4

    def test_segment_endpoint_is_deleted_second(self, view: ImageCanvas) -> None:
        self._prepare(view)
        view._segment_graphics[0].handles[0].setSelected(True)
        view._handles[0].setSelected(True)

        event = _key_press(view, Qt.Key_Delete)

        assert event.isAccepted() is True
        assert view.has_angle_measurements() is True
        assert view.has_segment_measurements() is False
        assert len(view.contour_points()) == 4

    def test_contour_nodes_are_deleted_last(self, view: ImageCanvas) -> None:
        self._prepare(view)
        view._handles[0].setSelected(True)

        event = _key_press(view, Qt.Key_Backspace)

        assert event.isAccepted() is True
        assert view.has_angle_measurements() is True
        assert view.has_segment_measurements() is True
        assert len(view.contour_points()) == 3

    def test_delete_without_selection_is_passed_through(
        self, view: ImageCanvas
    ) -> None:
        view.set_contour(SQUARE)

        event = _key_press(view, Qt.Key_Delete)

        assert event.isAccepted() is False


class TestKeyPressZoom:
    @pytest.mark.parametrize("key", [Qt.Key_Plus, Qt.Key_Equal])
    def test_zoom_in_keys(
        self, view: ImageCanvas, messages: list[str], key: int
    ) -> None:
        event = _key_press(view, key)

        assert _scale(view) == pytest.approx(1.2)
        assert event.isAccepted() is True
        assert messages == ["Масштаб увеличен."]

    def test_zoom_out_key(self, view: ImageCanvas, messages: list[str]) -> None:
        event = _key_press(view, Qt.Key_Minus)

        assert _scale(view) == pytest.approx(1 / 1.2)
        assert event.isAccepted() is True
        assert messages == ["Масштаб уменьшен."]

    def test_reset_zoom_key(self, view: ImageCanvas, messages: list[str]) -> None:
        view.scale(3.0, 3.0)

        event = _key_press(view, Qt.Key_0)

        assert _scale(view) == pytest.approx(1.0)
        assert event.isAccepted() is True
        assert messages == ["Масштаб: 100%."]


class TestWheelEvent:
    def test_ctrl_wheel_up_zooms_in(
        self, view: ImageCanvas, messages: list[str]
    ) -> None:
        event = _wheel(view, 120)

        assert _scale(view) == pytest.approx(1.2)
        assert event.isAccepted() is True
        assert messages == ["Масштаб увеличен."]

    def test_ctrl_wheel_down_zooms_out(
        self, view: ImageCanvas, messages: list[str]
    ) -> None:
        event = _wheel(view, -120)

        assert _scale(view) == pytest.approx(1 / 1.2)
        assert messages == ["Масштаб уменьшен."]

    def test_wheel_without_ctrl_scrolls(
        self, view: ImageCanvas, messages: list[str]
    ) -> None:
        _wheel(view, 120, modifiers=Qt.NoModifier)

        assert _scale(view) == pytest.approx(1.0)
        assert messages == []


class TestMouseDoubleClick:
    def test_node_is_inserted_near_segment(
        self, view: ImageCanvas, messages: list[str]
    ) -> None:
        view.set_contour(SQUARE)

        event = _double_click(view, QPointF(50, 12))

        assert len(view.contour_points()) == 5
        assert event.isAccepted() is True
        assert "Новый узел добавлен." in messages

    @pytest.mark.parametrize(
        "start_capture",
        ["begin_angle_measurement", "begin_segment_measurement", "begin_calibration"],
    )
    def test_ignored_during_capture(
        self, view: ImageCanvas, start_capture: str
    ) -> None:
        view.set_contour(SQUARE)
        getattr(view, start_capture)()

        event = _double_click(view, QPointF(50, 12))

        assert len(view.contour_points()) == 4
        assert event.isAccepted() is True

    def test_ignored_on_node_handle(self, view: ImageCanvas) -> None:
        view.set_contour(SQUARE)
        handle_position = view.mapFromScene(QPointF(10, 10))
        assert isinstance(view.itemAt(handle_position), NodeHandleItem)

        _double_click(view, QPointF(10, 10))

        assert len(view.contour_points()) == 4

    def test_ignored_when_contour_is_hidden(self, view: ImageCanvas) -> None:
        view.set_contour(SQUARE)
        view.set_contour_visible(False)

        _double_click(view, QPointF(50, 12))

        assert len(view.contour_points()) == 4

    def test_ignored_outside_image(self, view: ImageCanvas) -> None:
        view.set_contour(SQUARE)

        _double_click(view, QPointF(150, 120))

        assert len(view.contour_points()) == 4


class TestRemoveNode:
    def test_refuses_to_go_below_three_nodes(
        self,
        view: ImageCanvas,
        messages: list[str],
    ) -> None:
        view.set_contour(TRIANGLE)
        messages.clear()

        assert view.remove_node(0) is False
        assert messages == ["Контур должен содержать минимум 3 узла."]
        assert len(view.contour_points()) == 3

    @pytest.mark.parametrize("index", [-1, 4, 99])
    def test_index_out_of_range(self, view: ImageCanvas, index: int) -> None:
        view.set_contour(SQUARE)

        assert view.remove_node(index) is False
        assert len(view.contour_points()) == 4

    def test_valid_index_removes_node(
        self, view: ImageCanvas, messages: list[str]
    ) -> None:
        view.set_contour(SQUARE)
        messages.clear()

        assert view.remove_node(1) is True
        assert len(view.contour_points()) == 3
        assert messages[-1] == "Узел удалён."


class TestDeleteSelectedNodes:
    def test_nothing_selected(self, view: ImageCanvas) -> None:
        view.set_contour(SQUARE)

        assert view.delete_selected_nodes() is False

    def test_nothing_removable(self, view: ImageCanvas, messages: list[str]) -> None:
        view.set_contour(TRIANGLE)
        view._handles[0].setSelected(True)
        messages.clear()

        assert view.delete_selected_nodes() is False
        assert messages == [
            "Нельзя удалить больше узлов: контур должен остаться замкнутым."
        ]

    def test_removes_only_what_keeps_contour_closed(
        self,
        view: ImageCanvas,
        messages: list[str],
    ) -> None:
        view.set_contour(SQUARE)
        for handle in view._handles:
            handle.setSelected(True)
        messages.clear()

        assert view.delete_selected_nodes() is True
        assert len(view.contour_points()) == 3
        assert messages[-1] == "Удалено узлов: 1."

    def test_removes_all_selected_when_allowed(
        self,
        view: ImageCanvas,
        messages: list[str],
    ) -> None:
        view.set_contour([*SQUARE, Point(50, 75), Point(30, 75)])
        view._handles[0].setSelected(True)
        view._handles[3].setSelected(True)
        messages.clear()

        assert view.delete_selected_nodes() is True
        assert len(view.contour_points()) == 4
        assert messages[-1] == "Удалено узлов: 2."


class TestInsertNodeNear:
    def test_without_contour(self, view: ImageCanvas) -> None:
        assert view.insert_node_near(QPointF(50, 12)) is False

    def test_click_too_far_from_any_segment(
        self,
        view: ImageCanvas,
        messages: list[str],
    ) -> None:
        view.set_contour(SQUARE)
        messages.clear()

        assert view.insert_node_near(QPointF(50, 40)) is False
        assert messages == ["Щёлкните ближе к сегменту, чтобы добавить узел."]

    def test_tolerance_scales_with_zoom(self, view: ImageCanvas) -> None:
        view.set_contour(SQUARE)
        assert view._segment_pick_tolerance() == pytest.approx(14.0)

        view.scale(2.0, 2.0)
        assert view._segment_pick_tolerance() == pytest.approx(7.0)

        # На двойном увеличении точка в 10 px от сегмента уже вне допуска.
        assert view.insert_node_near(QPointF(50, 20)) is False

        view.resetTransform()
        assert view.insert_node_near(QPointF(50, 20)) is True

    def test_inserted_node_is_clamped_to_image(self, view: ImageCanvas) -> None:
        view.set_contour(SQUARE)

        assert view.insert_node_near(QPointF(50, 4)) is True
        inserted = view.contour_points()[1]
        assert 0 <= inserted.x <= 100
        assert 0 <= inserted.y <= 80


class TestSetContour:
    def test_requires_loaded_image(self, canvas: ImageCanvas) -> None:
        with pytest.raises(ValueError, match="без загруженного изображения"):
            canvas.set_contour(SQUARE)

    def test_requires_at_least_three_points(self, view: ImageCanvas) -> None:
        with pytest.raises(ValueError, match="минимум 3 точки"):
            view.set_contour([Point(1, 1), Point(2, 2)])

    def test_points_are_clamped_to_image_bounds(self, view: ImageCanvas) -> None:
        view.set_contour([Point(-50, -50), Point(500, 10), Point(50, 500)])

        points = view.contour_points()
        assert (points[0].x, points[0].y) == (0.0, 0.0)
        assert points[1].x == 100.0
        assert points[2].y == 80.0


class TestConstrainPoint:
    def test_clamps_to_image_rectangle(self, view: ImageCanvas) -> None:
        assert view.constrain_point(QPointF(-10, 200)) == QPointF(0.0, 80.0)
        assert view.constrain_point(QPointF(500, -7)) == QPointF(100.0, 0.0)
        assert view.constrain_point(QPointF(33, 44)) == QPointF(33.0, 44.0)

    def test_without_image_the_point_is_unchanged(self, canvas: ImageCanvas) -> None:
        assert canvas.constrain_point(QPointF(-10, 200)) == QPointF(-10.0, 200.0)


class TestZoom:
    def test_zoom_out_updates_labels_and_message(
        self,
        view: ImageCanvas,
        messages: list[str],
    ) -> None:
        view.set_calibration(Point(10, 10), Point(50, 10), "20 мм")
        view.set_segment_measurements([[Point(10, 30), Point(50, 30)]])
        messages.clear()

        view.zoom_out()

        assert _scale(view) == pytest.approx(1 / 1.2)
        assert messages == ["Масштаб уменьшен."]
        assert view._calibration_label_item.isVisible() is True

    def test_reset_zoom_recenters_on_image(
        self,
        view: ImageCanvas,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        centered: list[object] = []
        monkeypatch.setattr(
            type(view), "centerOn", lambda _self, item: centered.append(item)
        )
        view.scale(4.0, 4.0)

        view.reset_zoom()

        assert _scale(view) == pytest.approx(1.0)
        assert centered == [view._image_item]

    def test_reset_zoom_without_image(self, canvas: ImageCanvas, qapp) -> None:
        collected: list[str] = []
        canvas.message_changed.connect(collected.append)
        canvas.scale(2.0, 2.0)

        canvas.reset_zoom()

        assert canvas.transform().m11() == pytest.approx(1.0)
        assert collected == ["Масштаб: 100%."]

    def test_fit_to_image(self, view: ImageCanvas, messages: list[str]) -> None:
        view.resize(200, 200)
        messages.clear()

        view.fit_to_image()

        assert messages == ["Изображение вписано в окно."]
        assert _scale(view) != pytest.approx(1.0)

    def test_fit_to_image_without_image_is_noop(self, canvas: ImageCanvas) -> None:
        collected: list[str] = []
        canvas.message_changed.connect(collected.append)

        canvas.fit_to_image()

        assert collected == []


class TestMousePressGuards:
    @pytest.mark.parametrize(
        ("start_capture", "expected_message"),
        [
            ("begin_segment_measurement", "Отрезок: укажите точку внутри изображения."),
            ("begin_angle_measurement", "Угол: укажите точку внутри изображения."),
            ("begin_calibration", "Калибровка: укажите точку внутри изображения."),
        ],
    )
    def test_click_outside_image_is_rejected(
        self,
        view: ImageCanvas,
        messages: list[str],
        start_capture: str,
        expected_message: str,
    ) -> None:
        getattr(view, start_capture)()
        messages.clear()

        event = _mouse_press(view, QPointF(400, 400))

        assert messages == [expected_message]
        assert event.isAccepted() is True
        assert view._segment_capture_points == []
        assert view._angle_capture_points == []
        assert view._calibration_capture_points == []

    @pytest.mark.parametrize(
        ("start_capture", "flag", "expected_message"),
        [
            (
                "begin_segment_measurement",
                "_segment_capture_active",
                "Измерение отрезка отменено.",
            ),
            (
                "begin_angle_measurement",
                "_angle_capture_active",
                "Измерение угла отменено.",
            ),
            (
                "begin_calibration",
                "_calibration_capture_active",
                "Калибровка отменена.",
            ),
        ],
    )
    def test_right_button_cancels_capture(
        self,
        view: ImageCanvas,
        messages: list[str],
        start_capture: str,
        flag: str,
        expected_message: str,
    ) -> None:
        getattr(view, start_capture)()
        messages.clear()

        event = _mouse_press(view, QPointF(50, 40), button=Qt.RightButton)

        assert getattr(view, flag) is False
        assert event.isAccepted() is True
        assert expected_message in messages
        assert view.dragMode() == QGraphicsView.ScrollHandDrag

    def test_degenerate_angle_is_rejected_and_rolls_back(
        self,
        view: ImageCanvas,
        messages: list[str],
    ) -> None:
        view.begin_angle_measurement()
        _mouse_press(view, QPointF(20, 20))
        _mouse_press(view, QPointF(20, 40))
        messages.clear()

        event = _mouse_press(view, QPointF(20, 40))

        assert messages == [
            "Угол некорректен: точки лучей должны отличаться от вершины."
        ]
        assert event.isAccepted() is True
        assert len(view._angle_capture_points) == 2
        assert view.has_angle_measurements() is False


class TestCanvasState:
    def test_replace_current_rgb_array_keeps_contour_and_measurements(
        self,
        view: ImageCanvas,
    ) -> None:
        view.set_contour(SQUARE)
        view.set_angle_measurements([[Point(20, 20), Point(20, 40), Point(40, 40)]])
        view.set_segment_measurements([[Point(60, 20), Point(80, 40)]])
        view.set_calibration(Point(10, 10), Point(50, 10), "20 мм")

        new_array = np.full((80, 100, 3), 200, dtype=np.uint8)
        view.replace_current_rgb_array(new_array)

        assert np.array_equal(view.current_rgb_array()[0, 0], [200, 200, 200])
        assert len(view.contour_points()) == 4
        assert view.has_angle_measurements() is True
        assert view.has_segment_measurements() is True
        assert view.has_calibration() is True

    def test_replace_current_rgb_array_requires_image(
        self, canvas: ImageCanvas
    ) -> None:
        with pytest.raises(ValueError, match="Сначала загрузите изображение"):
            canvas.replace_current_rgb_array(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_contour_mask_covers_polygon(self, view: ImageCanvas) -> None:
        view.set_contour(SQUARE)

        mask = view._contour_mask()

        assert mask.shape == (80, 100)
        assert mask[40, 50] == 255
        assert mask[2, 2] == 0

    def test_contour_mask_is_empty_without_contour(self, view: ImageCanvas) -> None:
        assert view._contour_mask().any() == np.False_

    def test_contour_mask_requires_image(self, canvas: ImageCanvas) -> None:
        with pytest.raises(ValueError, match="Сначала загрузите изображение"):
            canvas._contour_mask()

    def test_flatten_background_to_white(
        self, view: ImageCanvas, messages: list[str]
    ) -> None:
        view.replace_current_rgb_array(np.full((80, 100, 3), 60, dtype=np.uint8))
        view.set_contour(SQUARE)
        messages.clear()

        view.flatten_background_to_white()

        rgb_array = view.current_rgb_array()
        assert tuple(rgb_array[40, 50]) == (60, 60, 60)
        assert tuple(rgb_array[2, 2]) == (255, 255, 255)
        assert "Фон за пределами контура выровнен до белого." in messages

    def test_flatten_background_requires_image(self, canvas: ImageCanvas) -> None:
        with pytest.raises(ValueError, match="Сначала загрузите изображение"):
            canvas.flatten_background_to_white()

    def test_flatten_background_requires_contour(self, view: ImageCanvas) -> None:
        with pytest.raises(ValueError, match="Сначала постройте или загрузите контур"):
            view.flatten_background_to_white()

    def test_clear_image_resets_everything(self, view: ImageCanvas) -> None:
        view.set_contour(SQUARE)
        view.set_angle_measurements([[Point(20, 20), Point(20, 40), Point(40, 40)]])
        view.set_segment_measurements([[Point(60, 20), Point(80, 40)]])
        view.set_calibration(Point(10, 10), Point(50, 10), "20 мм")
        image_states: list[bool] = []
        view.image_state_changed.connect(image_states.append)

        view.clear_image()

        assert view.has_image() is False
        assert view.has_contour() is False
        assert view.has_calibration() is False
        assert view.has_angle_measurements() is False
        assert view.has_segment_measurements() is False
        assert view.image_size() is None
        assert view.current_image_path() is None
        assert image_states == [False]

    def test_clear_contour_emits_state_change(self, view: ImageCanvas) -> None:
        view.set_contour(SQUARE)
        states: list[bool] = []
        view.contour_state_changed.connect(states.append)

        view.clear_contour()

        assert states == [False]
        assert view.contour_points() == []
        assert view.contour_line_color() == "#0b84c6"
        assert view.is_contour_visible() is False

    def test_clear_angles_emits_state_change(self, view: ImageCanvas) -> None:
        view.set_angle_measurements([[Point(20, 20), Point(20, 40), Point(40, 40)]])
        states: list[int] = []
        view.angle_state_changed.connect(lambda: states.append(1))

        view.clear_angles()

        assert view.has_angle_measurements() is False
        assert view._angle_graphics == []
        assert states == [1]

    def test_clear_segments_emits_state_change(self, view: ImageCanvas) -> None:
        view.set_segment_measurements([[Point(60, 20), Point(80, 40)]])
        states: list[int] = []
        view.segment_state_changed.connect(lambda: states.append(1))

        view.clear_segments()

        assert view.has_segment_measurements() is False
        assert view._segment_graphics == []
        assert states == [1]


class TestMeasurementLookupsWithUnknownId:
    @pytest.fixture()
    def measured(self, view: ImageCanvas) -> ImageCanvas:
        view.set_angle_measurements([[Point(20, 20), Point(20, 40), Point(40, 40)]])
        view.set_segment_measurements([[Point(60, 20), Point(80, 40)]])
        return view

    def test_set_names_with_unknown_id(self, measured: ImageCanvas) -> None:
        assert measured.set_angle_measurement_name("нет-такого", "Имя") is False
        assert measured.set_segment_measurement_name("нет-такого", "Имя") is False
        assert measured.set_segment_labels("нет-такого", "A", "B") is False

    def test_set_colors_with_unknown_id(self, measured: ImageCanvas) -> None:
        assert (
            measured.set_angle_measurement_line_color("нет-такого", "#123456") is False
        )
        assert (
            measured.set_angle_measurement_label_color("нет-такого", "#123456") is False
        )
        assert (
            measured.set_segment_measurement_line_color("нет-такого", "#123456")
            is False
        )
        assert (
            measured.set_segment_measurement_label_color("нет-такого", "#123456")
            is False
        )

    def test_read_colors_with_unknown_id(self, measured: ImageCanvas) -> None:
        assert measured.angle_measurement_line_color("нет-такого") is None
        assert measured.angle_measurement_label_color("нет-такого") is None
        assert measured.segment_measurement_line_color("нет-такого") is None
        assert measured.segment_measurement_label_color("нет-такого") is None

    def test_invalid_color_falls_back_to_default_and_reports_no_change(
        self,
        measured: ImageCanvas,
    ) -> None:
        angle_id = measured.angle_measurement_records()[0][0]
        segment_id = measured.segment_measurement_records()[0][0]

        assert measured.set_angle_measurement_line_color(angle_id, "не цвет") is False
        assert measured.angle_measurement_line_color(angle_id) == ANGLE_LINE_COLOR

        assert measured.set_segment_measurement_label_color(segment_id, "#xyz") is False
        assert (
            measured.segment_measurement_label_color(segment_id) == SEGMENT_LABEL_COLOR
        )

    def test_visibility_toggles_with_unknown_id(self, measured: ImageCanvas) -> None:
        assert measured.set_angle_measurement_visible("нет-такого", False) is False
        assert measured.set_segment_measurement_visible("нет-такого", False) is False
        assert measured.is_angle_measurement_visible("нет-такого") is False
        assert measured.is_segment_measurement_visible("нет-такого") is False

    def test_setting_the_same_value_reports_no_change(
        self, measured: ImageCanvas
    ) -> None:
        angle_id = measured.angle_measurement_records()[0][0]

        assert measured.set_angle_measurement_name(angle_id, "Первый") is True
        assert measured.set_angle_measurement_name(angle_id, "Первый") is False


class TestDistanceToSegment:
    def test_zero_length_segment(self) -> None:
        distance, projection = _distance_to_segment(
            QPointF(3, 4), QPointF(0, 0), QPointF(0, 0)
        )

        assert distance == pytest.approx(5.0)
        assert projection == QPointF(0.0, 0.0)

    def test_projection_inside_segment(self) -> None:
        distance, projection = _distance_to_segment(
            QPointF(5, 3), QPointF(0, 0), QPointF(10, 0)
        )

        assert distance == pytest.approx(3.0)
        assert projection == QPointF(5.0, 0.0)

    def test_projection_is_clamped_to_segment_ends(self) -> None:
        distance, projection = _distance_to_segment(
            QPointF(-5, 0), QPointF(0, 0), QPointF(10, 0)
        )

        assert distance == pytest.approx(5.0)
        assert projection == QPointF(0.0, 0.0)


class TestCalibrationLengthFromText:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("12,5", 12.5),
            ("12.5", 12.5),
            ("12,5 мм", 12.5),
            ("12.5mm", 12.5),
            ("  7 ММ ", 7.0),
        ],
    )
    def test_accepted_values(self, text: str, expected: float) -> None:
        assert _calibration_length_mm_from_text(text) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "text", ["", "около десяти", "-5", "0", "-0.001", "мм", None]
    )
    def test_rejected_values(self, text: str | None) -> None:
        assert _calibration_length_mm_from_text(text) is None


class TestCompactFloat:
    @pytest.mark.parametrize(
        ("value", "decimals", "expected"),
        [
            (12.500, 2, "12.5"),
            (12.0, 2, "12"),
            (0.0, 2, "0"),
            (0.004, 2, "0"),
            (3.14159, 3, "3.142"),
            (-2.50, 1, "-2.5"),
        ],
    )
    def test_trailing_zeros_are_trimmed(
        self,
        value: float,
        decimals: int,
        expected: str,
    ) -> None:
        assert _compact_float(value, decimals) == expected


class TestFormatAngleDegrees:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (90.0, "90°"),
            (89.96, "90°"),
            (45.25, "45.2°"),
            (45.26, "45.3°"),
            (0.0, "0°"),
        ],
    )
    def test_rounding_and_format(self, value: float, expected: str) -> None:
        assert _format_angle_degrees(value) == expected


class TestAngleArcPath:
    def test_degenerate_ray_returns_empty_path(self) -> None:
        path = _angle_arc_path(QPointF(10, 10), QPointF(10, 10), QPointF(20, 10))

        assert path.isEmpty() is True

    def test_overlapping_rays_return_empty_path(self) -> None:
        path = _angle_arc_path(QPointF(10, 0), QPointF(0, 0), QPointF(20, 0))

        assert path.isEmpty() is True

    def test_straight_angle_produces_half_circle(self) -> None:
        path = _angle_arc_path(QPointF(0, 0), QPointF(10, 0), QPointF(20, 0))

        assert path.isEmpty() is False

    def test_right_angle_produces_arc(self) -> None:
        path = _angle_arc_path(QPointF(10, 0), QPointF(0, 0), QPointF(0, 10))

        assert path.isEmpty() is False
        assert path.elementCount() > 8

    def test_arc_radius_is_bounded_by_shortest_ray(self) -> None:
        path = _angle_arc_path(QPointF(3, 0), QPointF(0, 0), QPointF(0, 100))
        bounding = path.boundingRect()

        assert max(bounding.width(), bounding.height()) <= 2 * 3 + 1e-6


def test_canvas_without_image_reports_empty_pixels(canvas: ImageCanvas) -> None:
    assert canvas.contour_rgb_pixels().shape == (0, 3)


def test_canvas_pixels_inside_contour(view: ImageCanvas) -> None:
    view.replace_current_rgb_array(np.full((80, 100, 3), 42, dtype=np.uint8))
    view.set_contour(SQUARE)

    pixels = view.contour_rgb_pixels()

    assert pixels.shape[1] == 3
    assert pixels.shape[0] > 0
    assert (pixels == 42).all()


def test_set_loaded_image_resets_previous_state(view: ImageCanvas) -> None:
    view.set_contour(SQUARE)
    view.set_angle_measurements([[Point(20, 20), Point(20, 40), Point(40, 40)]])

    view.set_loaded_image(
        loaded_image_from_rgb_array(
            Path("другой.png"), np.zeros((40, 50, 3), dtype=np.uint8)
        )
    )

    assert view.image_size() == (50, 40)
    assert view.has_contour() is False
    assert view.has_angle_measurements() is False


def test_angle_helpers_agree_with_math() -> None:
    assert _format_angle_degrees(math.degrees(math.pi / 2)) == "90°"
