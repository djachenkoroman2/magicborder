from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QKeyEvent, QPainter, QPainterPath, QPen, QPixmap, QWheelEvent
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from .io_utils import LoadedImage, loaded_image_from_rgb_array
from .models import Point


class NodeHandleItem(QGraphicsEllipseItem):
    def __init__(self, canvas: "ImageCanvas", index: int, position: QPointF) -> None:
        radius = 5.5
        super().__init__(-radius, -radius, radius * 2.0, radius * 2.0)
        self.canvas = canvas
        self.index = index
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setBrush(QColor("#ffb000"))
        self.setPen(QPen(QColor("white"), 1.4))
        self.setZValue(20)
        self.setToolTip("Перетащите для редактирования. Правая кнопка мыши удаляет узел.")
        self.setPos(position)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: object) -> object:
        if change == QGraphicsItem.ItemPositionChange and isinstance(value, QPointF):
            return self.canvas.constrain_point(value)
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.canvas.handle_moved(self.index, self.pos())
        return super().itemChange(change, value)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self.canvas.remove_node(self.index)
            event.accept()
            return
        super().mousePressEvent(event)


class CalibrationHandleItem(QGraphicsEllipseItem):
    def __init__(self, canvas: "ImageCanvas", index: int, position: QPointF) -> None:
        radius = 6.0
        super().__init__(-radius, -radius, radius * 2.0, radius * 2.0)
        self.canvas = canvas
        self.index = index
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setBrush(QColor("#2a9d8f"))
        self.setPen(QPen(QColor("white"), 1.5))
        self.setZValue(26)
        self.setToolTip("Перетащите для редактирования калибровочного отрезка.")
        self.setPos(position)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: object) -> object:
        if change == QGraphicsItem.ItemPositionChange and isinstance(value, QPointF):
            return self.canvas.constrain_point(value)
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.canvas.calibration_handle_moved(self.index, self.pos())
        return super().itemChange(change, value)


class ImageCanvas(QGraphicsView):
    message_changed = pyqtSignal(str)
    image_state_changed = pyqtSignal(bool)
    contour_state_changed = pyqtSignal(bool)
    contour_geometry_changed = pyqtSignal()
    calibration_segment_selected = pyqtSignal(object)
    calibration_geometry_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._image_item = QGraphicsPixmapItem()
        self._image_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._image_item)

        self._path_item = QGraphicsPathItem()
        contour_pen = QPen(QColor("#0b84c6"), 2.0)
        contour_pen.setCosmetic(True)
        self._path_item.setPen(contour_pen)
        self._path_item.setBrush(QBrush(Qt.NoBrush))
        self._path_item.setZValue(10)
        self._scene.addItem(self._path_item)

        self._calibration_line_item = QGraphicsLineItem()
        calibration_pen = QPen(QColor("#d9467f"), 2.0)
        calibration_pen.setCosmetic(True)
        calibration_pen.setStyle(Qt.DashLine)
        self._calibration_line_item.setPen(calibration_pen)
        self._calibration_line_item.setZValue(16)
        self._calibration_line_item.setVisible(False)
        self._scene.addItem(self._calibration_line_item)

        self._loaded_image: LoadedImage | None = None
        self._contour_points: list[QPointF] = []
        self._handles: list[NodeHandleItem] = []
        self._suppress_handle_events = False
        self._calibration_points: list[QPointF] = []
        self._calibration_handles: list[CalibrationHandleItem] = []
        self._suppress_calibration_handle_events = False
        self._calibration_capture_active = False
        self._calibration_capture_points: list[QPointF] = []

        self.setRenderHints(
            QPainter.Antialiasing
            | QPainter.SmoothPixmapTransform
            | QPainter.TextAntialiasing
        )
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#eef2f7"))

    def has_image(self) -> bool:
        return self._loaded_image is not None

    def has_contour(self) -> bool:
        return len(self._contour_points) >= 3

    def has_calibration(self) -> bool:
        return len(self._calibration_points) == 2

    def current_image_path(self) -> Path | None:
        return self._loaded_image.path if self._loaded_image else None

    def image_size(self) -> tuple[int, int] | None:
        if not self._loaded_image:
            return None
        return self._loaded_image.width, self._loaded_image.height

    def current_rgb_array(self) -> np.ndarray:
        if not self._loaded_image:
            raise ValueError("Сначала загрузите изображение проекта.")
        return self._loaded_image.rgb_array.copy()

    def contour_points(self) -> list[Point]:
        return [Point(point.x(), point.y()) for point in self._contour_points]

    def calibration_points(self) -> list[Point]:
        return [Point(point.x(), point.y()) for point in self._calibration_points]

    def contour_rgb_pixels(self) -> np.ndarray:
        if not self._loaded_image or len(self._contour_points) < 3:
            return np.empty((0, 3), dtype=np.uint8)

        mask = self._contour_mask()
        pixels = self._loaded_image.rgb_array[mask > 0]
        return np.ascontiguousarray(pixels.reshape((-1, 3)))

    def set_loaded_image(self, image: LoadedImage) -> None:
        self._loaded_image = image
        self._image_item.setPixmap(image.pixmap)
        self._image_item.setOffset(0, 0)
        self._scene.setSceneRect(QRectF(0, 0, image.width, image.height))
        self.clear_contour()
        self.clear_calibration()
        self.fit_to_image()
        self.image_state_changed.emit(True)
        self.message_changed.emit(f"Открыто изображение: {image.path.name}")

    def clear_image(self) -> None:
        self._loaded_image = None
        self._image_item.setPixmap(QPixmap())
        self._scene.setSceneRect(QRectF())
        self.clear_contour()
        self.clear_calibration()
        self.resetTransform()
        self.image_state_changed.emit(False)
        self.message_changed.emit("Изображение очищено.")

    def replace_current_rgb_array(self, rgb_array: np.ndarray) -> None:
        if not self._loaded_image:
            raise ValueError("Сначала загрузите изображение проекта.")
        updated_image = loaded_image_from_rgb_array(self._loaded_image.path, rgb_array)
        self._loaded_image = updated_image
        self._image_item.setPixmap(updated_image.pixmap)
        self._image_item.setOffset(0, 0)
        self._scene.setSceneRect(QRectF(0, 0, updated_image.width, updated_image.height))
        self._refresh_path()
        self._rebuild_handles()
        self.image_state_changed.emit(True)

    def flatten_background_to_white(self) -> None:
        if not self._loaded_image:
            raise ValueError("Сначала загрузите изображение проекта.")
        if len(self._contour_points) < 3:
            raise ValueError("Сначала постройте или загрузите контур.")

        rgb_array = self.current_rgb_array()
        mask = self._contour_mask()
        rgb_array[mask == 0] = 255
        self.replace_current_rgb_array(rgb_array)
        self.message_changed.emit("Фон за пределами контура выровнен до белого.")

    def clear_contour(self) -> None:
        self._contour_points.clear()
        self._path_item.setPath(QPainterPath())
        self._clear_handles()
        self.contour_state_changed.emit(False)
        self.contour_geometry_changed.emit()

    def set_contour(self, points: Sequence[Point]) -> None:
        if not self._loaded_image:
            raise ValueError("Нельзя задать контур без загруженного изображения.")
        if len(points) < 3:
            raise ValueError("Контур должен содержать минимум 3 точки.")

        self._contour_points = [
            self.constrain_point(QPointF(float(point.x), float(point.y)))
            for point in points
        ]
        self._refresh_path()
        self._rebuild_handles()
        self.contour_state_changed.emit(True)
        self.contour_geometry_changed.emit()
        self.message_changed.emit(f"Контур загружен: {len(self._contour_points)} узлов.")

    def begin_calibration(self) -> None:
        if not self.has_image():
            self.message_changed.emit("Сначала выберите изображение проекта.")
            return
        self._calibration_capture_active = True
        self._calibration_capture_points.clear()
        self.setDragMode(QGraphicsView.NoDrag)
        self.message_changed.emit("Калибровка: укажите начало отрезка.")

    def cancel_calibration(self) -> None:
        if not self._calibration_capture_active:
            return
        self._calibration_capture_active = False
        self._calibration_capture_points.clear()
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.message_changed.emit("Калибровка отменена.")

    def set_calibration(self, start: Point, end: Point) -> None:
        if not self._loaded_image:
            raise ValueError("Нельзя задать калибровку без загруженного изображения.")
        self._calibration_capture_active = False
        self._calibration_capture_points.clear()
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._calibration_points = [
            self.constrain_point(QPointF(float(start.x), float(start.y))),
            self.constrain_point(QPointF(float(end.x), float(end.y))),
        ]
        self._refresh_calibration_line()
        self._rebuild_calibration_handles()
        self.message_changed.emit("Калибровочный отрезок задан.")

    def clear_calibration(self) -> None:
        self._calibration_capture_active = False
        self._calibration_capture_points.clear()
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._calibration_points.clear()
        self._refresh_calibration_line()
        self._clear_calibration_handles()

    def constrain_point(self, point: QPointF) -> QPointF:
        if not self._loaded_image:
            return QPointF(point)
        rect = self._image_item.boundingRect()
        clamped_x = min(max(point.x(), rect.left()), rect.right())
        clamped_y = min(max(point.y(), rect.top()), rect.bottom())
        return QPointF(clamped_x, clamped_y)

    def handle_moved(self, index: int, position: QPointF) -> None:
        if self._suppress_handle_events or index >= len(self._contour_points):
            return
        self._contour_points[index] = self.constrain_point(position)
        self._refresh_path()
        self.contour_geometry_changed.emit()

    def calibration_handle_moved(self, index: int, position: QPointF) -> None:
        if self._suppress_calibration_handle_events or index >= len(self._calibration_points):
            return
        self._calibration_points[index] = self.constrain_point(position)
        self._refresh_calibration_line()
        self.calibration_geometry_changed.emit()

    def remove_node(self, index: int) -> bool:
        if len(self._contour_points) <= 3:
            self.message_changed.emit("Контур должен содержать минимум 3 узла.")
            return False
        if not (0 <= index < len(self._contour_points)):
            return False

        del self._contour_points[index]
        self._refresh_path()
        self._rebuild_handles()
        self.contour_state_changed.emit(True)
        self.contour_geometry_changed.emit()
        self.message_changed.emit("Узел удалён.")
        return True

    def delete_selected_nodes(self) -> bool:
        selected_indexes = sorted(
            {handle.index for handle in self._handles if handle.isSelected()},
            reverse=True,
        )
        if not selected_indexes:
            return False

        removable_count = max(0, len(self._contour_points) - 3)
        if removable_count == 0:
            self.message_changed.emit("Нельзя удалить больше узлов: контур должен остаться замкнутым.")
            return False

        removed = 0
        for index in selected_indexes[:removable_count]:
            del self._contour_points[index]
            removed += 1

        if removed == 0:
            return False

        self._refresh_path()
        self._rebuild_handles()
        self.contour_state_changed.emit(True)
        self.contour_geometry_changed.emit()
        self.message_changed.emit(f"Удалено узлов: {removed}.")
        return True

    def insert_node_near(self, scene_pos: QPointF) -> bool:
        if not self.has_contour():
            return False

        best_distance = float("inf")
        best_projection = None
        best_insert_index = None

        for index, start in enumerate(self._contour_points):
            end = self._contour_points[(index + 1) % len(self._contour_points)]
            distance, projection = _distance_to_segment(scene_pos, start, end)
            if distance < best_distance:
                best_distance = distance
                best_projection = projection
                best_insert_index = index + 1

        if best_projection is None or best_insert_index is None:
            return False

        if best_distance > self._segment_pick_tolerance():
            self.message_changed.emit("Щёлкните ближе к сегменту, чтобы добавить узел.")
            return False

        self._contour_points.insert(best_insert_index, self.constrain_point(best_projection))
        self._refresh_path()
        self._rebuild_handles()
        self.contour_state_changed.emit(True)
        self.contour_geometry_changed.emit()
        self.message_changed.emit("Новый узел добавлен.")
        return True

    def zoom_in(self) -> None:
        self.scale(1.2, 1.2)
        self.message_changed.emit("Масштаб увеличен.")

    def zoom_out(self) -> None:
        self.scale(1 / 1.2, 1 / 1.2)
        self.message_changed.emit("Масштаб уменьшен.")

    def reset_zoom(self) -> None:
        self.resetTransform()
        if self.has_image():
            self.centerOn(self._image_item)
        self.message_changed.emit("Масштаб: 100%.")

    def fit_to_image(self) -> None:
        if not self.has_image():
            return
        self.fitInView(self._image_item, Qt.KeepAspectRatio)
        self.message_changed.emit("Изображение вписано в окно.")

    def mousePressEvent(self, event) -> None:
        if self._calibration_capture_active and event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if not self._image_item.boundingRect().contains(scene_pos):
                self.message_changed.emit("Калибровка: укажите точку внутри изображения.")
                event.accept()
                return

            self._calibration_capture_points.append(self.constrain_point(scene_pos))
            if len(self._calibration_capture_points) == 1:
                self.message_changed.emit("Калибровка: укажите конец отрезка.")
            else:
                points = [Point(point.x(), point.y()) for point in self._calibration_capture_points[:2]]
                self._calibration_capture_active = False
                self._calibration_capture_points.clear()
                self.setDragMode(QGraphicsView.ScrollHandDrag)
                self.calibration_segment_selected.emit(points)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.has_contour():
            item = self.itemAt(event.pos())
            if not isinstance(item, NodeHandleItem):
                scene_pos = self.mapToScene(event.pos())
                if self._image_item.boundingRect().contains(scene_pos):
                    if self.insert_node_near(scene_pos):
                        event.accept()
                        return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape and self._calibration_capture_active:
            self.cancel_calibration()
            event.accept()
            return
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.delete_selected_nodes():
            event.accept()
            return
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_in()
            event.accept()
            return
        if event.key() == Qt.Key_Minus:
            self.zoom_out()
            event.accept()
            return
        if event.key() == Qt.Key_0:
            self.reset_zoom()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
            return
        super().wheelEvent(event)

    def _segment_pick_tolerance(self) -> float:
        scale = abs(self.transform().m11()) or 1.0
        return 14.0 / scale

    def _refresh_path(self) -> None:
        path = QPainterPath()
        if self._contour_points:
            path.moveTo(self._contour_points[0])
            for point in self._contour_points[1:]:
                path.lineTo(point)
            path.closeSubpath()
        self._path_item.setPath(path)

    def _refresh_calibration_line(self) -> None:
        if len(self._calibration_points) != 2:
            self._calibration_line_item.setVisible(False)
            self._calibration_line_item.setLine(0.0, 0.0, 0.0, 0.0)
            return

        start, end = self._calibration_points
        self._calibration_line_item.setLine(start.x(), start.y(), end.x(), end.y())
        self._calibration_line_item.setVisible(True)

    def _clear_handles(self) -> None:
        for handle in self._handles:
            self._scene.removeItem(handle)
        self._handles.clear()

    def _rebuild_handles(self) -> None:
        self._clear_handles()
        self._suppress_handle_events = True
        try:
            for index, point in enumerate(self._contour_points):
                handle = NodeHandleItem(self, index, point)
                self._scene.addItem(handle)
                self._handles.append(handle)
        finally:
            self._suppress_handle_events = False

    def _clear_calibration_handles(self) -> None:
        for handle in self._calibration_handles:
            self._scene.removeItem(handle)
        self._calibration_handles.clear()

    def _rebuild_calibration_handles(self) -> None:
        self._clear_calibration_handles()
        self._suppress_calibration_handle_events = True
        try:
            for index, point in enumerate(self._calibration_points):
                handle = CalibrationHandleItem(self, index, point)
                self._scene.addItem(handle)
                self._calibration_handles.append(handle)
        finally:
            self._suppress_calibration_handle_events = False

    def _contour_mask(self) -> np.ndarray:
        if not self._loaded_image:
            raise ValueError("Сначала загрузите изображение проекта.")

        mask = np.zeros(self._loaded_image.rgb_array.shape[:2], dtype=np.uint8)
        if len(self._contour_points) < 3:
            return mask

        polygon = np.array(
            [
                [int(round(point.x())), int(round(point.y()))]
                for point in self._contour_points
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [polygon], 255)
        return mask


def _distance_to_segment(point: QPointF, start: QPointF, end: QPointF) -> tuple[float, QPointF]:
    start_x, start_y = start.x(), start.y()
    end_x, end_y = end.x(), end.y()
    point_x, point_y = point.x(), point.y()

    delta_x = end_x - start_x
    delta_y = end_y - start_y
    segment_length_squared = delta_x * delta_x + delta_y * delta_y
    if segment_length_squared <= 1e-9:
        projection = QPointF(start_x, start_y)
        distance = ((point_x - start_x) ** 2 + (point_y - start_y) ** 2) ** 0.5
        return distance, projection

    projection_factor = (
        ((point_x - start_x) * delta_x + (point_y - start_y) * delta_y)
        / segment_length_squared
    )
    projection_factor = max(0.0, min(1.0, projection_factor))

    projected_x = start_x + projection_factor * delta_x
    projected_y = start_y + projection_factor * delta_y
    distance = ((point_x - projected_x) ** 2 + (point_y - projected_y) ** 2) ** 0.5
    return distance, QPointF(projected_x, projected_y)
