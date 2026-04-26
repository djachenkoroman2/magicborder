from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .icons import load_icon


@dataclass(frozen=True, slots=True)
class HistogramSeries:
    name: str
    values: np.ndarray
    color: QColor


@dataclass(frozen=True, slots=True)
class HistogramPlotData:
    series: tuple[HistogramSeries, ...]
    x_ticks: tuple[tuple[int, str], ...]
    x_label: str
    sample_count: int


class HistogramPanel(QFrame):
    def __init__(
        self,
        title: str,
        default_file_name: str,
        parent: QWidget | None = None,
        *,
        default_file_name_provider: Callable[[], str] | None = None,
    ) -> None:
        super().__init__(parent)
        self._default_file_name = default_file_name
        self._default_file_name_provider = default_file_name_provider

        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("histogramPanel")
        self.setMinimumHeight(145)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        title_label = QLabel(title)
        title_label.setObjectName("histogramTitle")

        self._save_button = QToolButton()
        self._save_button.setIcon(load_icon("save-image"))
        self._save_button.setIconSize(QSize(20, 20))
        self._save_button.setToolTip(f"Сохранить {title.lower()} в PNG")
        self._save_button.setStatusTip(f"Сохранить текущий вид {title.lower()} на диск.")
        self._save_button.setAutoRaise(True)
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._save_histogram)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self._save_button)

        self.canvas = HistogramCanvas()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(header_layout)
        layout.addWidget(self.canvas, 1)

    def set_histogram(self, data: HistogramPlotData) -> None:
        self.canvas.set_plot_data(data)
        self._save_button.setEnabled(True)

    def clear_histogram(self, message: str) -> None:
        self.canvas.clear_plot_data(message)
        self._save_button.setEnabled(False)

    def _save_histogram(self) -> None:
        if not self.canvas.has_plot_data():
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить гистограмму",
            self.default_file_name(),
            "PNG image (*.png);;All files (*)",
        )
        if not file_name:
            return

        output_path = _ensure_png_suffix(Path(file_name))
        try:
            self.canvas.save_png(output_path)
        except ValueError as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))

    def default_file_name(self) -> str:
        if self._default_file_name_provider is None:
            return self._default_file_name
        return self._default_file_name_provider()


class HistogramCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plot_data: HistogramPlotData | None = None
        self._empty_message = "Создайте контур, чтобы увидеть гистограмму."
        self.setMinimumSize(220, 95)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(False)

    def has_plot_data(self) -> bool:
        return self._plot_data is not None

    def set_plot_data(self, data: HistogramPlotData) -> None:
        self._plot_data = data
        self.update()

    def clear_plot_data(self, message: str) -> None:
        self._plot_data = None
        self._empty_message = message
        self.update()

    def save_png(self, path: Path) -> None:
        pixmap = QPixmap(self.size())
        pixmap.fill(QColor("#171b20"))
        painter = QPainter(pixmap)
        try:
            self.render(painter)
        finally:
            painter.end()

        if not pixmap.save(str(path), "PNG"):
            raise ValueError("Не удалось сохранить гистограмму в PNG.")

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        canvas_rect = self.rect()
        painter.fillRect(canvas_rect, QColor("#171b20"))

        if self._plot_data is None:
            self._draw_empty_state(painter, QRectF(canvas_rect))
            return

        margins = self._plot_margins()
        plot_rect = QRectF(
            margins[0],
            margins[1],
            max(1, self.width() - margins[0] - margins[2]),
            max(1, self.height() - margins[1] - margins[3]),
        )

        self._draw_grid(painter, plot_rect, self._plot_data)
        self._draw_series(painter, plot_rect, self._plot_data.series)
        self._draw_axes_text(painter, plot_rect, self._plot_data)
        self._draw_legend(painter, self._plot_data.series)

    def _plot_margins(self) -> tuple[int, int, int, int]:
        if self.width() < 280 or self.height() < 180:
            return 36, 18, 10, 34
        return 48, 24, 14, 46

    def _draw_empty_state(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QColor("#9aa4b2"))
        painter.setFont(_font(9))
        text_rect = rect.adjusted(16, 16, -16, -16)
        painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, self._empty_message)

    def _draw_grid(
        self,
        painter: QPainter,
        plot_rect: QRectF,
        data: HistogramPlotData,
    ) -> None:
        painter.setPen(QPen(QColor("#2c333d"), 1.0))
        for step in range(6):
            y = plot_rect.top() + plot_rect.height() * step / 5
            painter.drawLine(QPointF(plot_rect.left(), y), QPointF(plot_rect.right(), y))

        for tick, _label in data.x_ticks:
            x = _x_for_bin(plot_rect, tick)
            painter.drawLine(QPointF(x, plot_rect.top()), QPointF(x, plot_rect.bottom()))

        painter.setPen(QPen(QColor("#6d7785"), 1.2))
        painter.drawRect(plot_rect)

    def _draw_series(
        self,
        painter: QPainter,
        plot_rect: QRectF,
        series: Iterable[HistogramSeries],
    ) -> None:
        max_value = max((int(item.values.max()) for item in series), default=0)
        if max_value <= 0:
            return

        for item in series:
            points = QPolygonF()
            for index, value in enumerate(item.values):
                x = _x_for_bin(plot_rect, index)
                y = plot_rect.bottom() - (float(value) / max_value) * plot_rect.height()
                points.append(QPointF(x, y))

            color = QColor(item.color)
            color.setAlpha(220)
            painter.setPen(QPen(color, 1.6))
            painter.drawPolyline(points)

    def _draw_axes_text(
        self,
        painter: QPainter,
        plot_rect: QRectF,
        data: HistogramPlotData,
    ) -> None:
        painter.setFont(_font(8))
        painter.setPen(QColor("#b7c0cc"))
        metrics = QFontMetrics(painter.font())

        max_value = max((int(item.values.max()) for item in data.series), default=0)
        painter.drawText(
            QRectF(2, plot_rect.top() - 8, plot_rect.left() - 8, 18),
            Qt.AlignRight | Qt.AlignVCenter,
            _short_count(max_value),
        )
        painter.drawText(
            QRectF(2, plot_rect.bottom() - 10, plot_rect.left() - 8, 18),
            Qt.AlignRight | Qt.AlignVCenter,
            "0",
        )

        for tick, label in data.x_ticks:
            x = _x_for_bin(plot_rect, tick)
            label_width = metrics.horizontalAdvance(label)
            label_rect = QRectF(x - label_width / 2, plot_rect.bottom() + 4, label_width + 2, 16)
            painter.drawText(label_rect, Qt.AlignCenter, label)

        axis_label = metrics.elidedText(data.x_label, Qt.ElideRight, max(60, int(plot_rect.width())))
        painter.drawText(
            QRectF(plot_rect.left(), self.height() - 20, plot_rect.width(), 16),
            Qt.AlignCenter,
            axis_label,
        )

        sample_text = f"Пикселей: {_short_count(data.sample_count)}"
        sample_width = metrics.horizontalAdvance(sample_text)
        sample_y = max(0.0, plot_rect.top() - 21.0)
        painter.drawText(
            QRectF(plot_rect.right() - sample_width, sample_y, sample_width, 16),
            Qt.AlignRight | Qt.AlignVCenter,
            sample_text,
        )

    def _draw_legend(self, painter: QPainter, series: tuple[HistogramSeries, ...]) -> None:
        painter.setFont(_font(8, bold=True))
        metrics = QFontMetrics(painter.font())
        x = 8
        y = 9
        for item in series:
            text_width = metrics.horizontalAdvance(item.name)
            if x + text_width + 24 > self.width() - 8:
                break

            painter.setPen(QPen(item.color, 2.4))
            painter.drawLine(QPointF(x, y + 7), QPointF(x + 14, y + 7))
            painter.setPen(QColor("#d7dde6"))
            painter.drawText(QRectF(x + 18, y, text_width + 2, 16), Qt.AlignLeft, item.name)
            x += text_width + 34


def build_rgb_histogram(rgb_pixels: np.ndarray) -> HistogramPlotData | None:
    if rgb_pixels.size == 0:
        return None

    channel_specs = (
        ("R", QColor("#ff5b63"), 0),
        ("G", QColor("#35c46f"), 1),
        ("B", QColor("#4aa3ff"), 2),
    )
    series = tuple(
        HistogramSeries(
            name=name,
            color=color,
            values=np.bincount(rgb_pixels[:, channel_index], minlength=256)[:256],
        )
        for name, color, channel_index in channel_specs
    )
    return HistogramPlotData(
        series=series,
        x_ticks=((0, "0"), (64, "64"), (128, "128"), (192, "192"), (255, "255")),
        x_label="Значение канала RGB, 0..255",
        sample_count=int(rgb_pixels.shape[0]),
    )


def build_lab_histogram(rgb_pixels: np.ndarray) -> HistogramPlotData | None:
    if rgb_pixels.size == 0:
        return None

    lab_pixels = cv2.cvtColor(rgb_pixels.reshape((-1, 1, 3)), cv2.COLOR_RGB2LAB).reshape((-1, 3))
    lab_pixels = lab_pixels.astype(np.float32)

    lightness = lab_pixels[:, 0] * (100.0 / 255.0)
    a_values = lab_pixels[:, 1] - 128.0
    b_values = lab_pixels[:, 2] - 128.0

    series = (
        HistogramSeries(
            name="L",
            color=QColor("#eef2f7"),
            values=np.histogram(lightness, bins=256, range=(0.0, 100.0))[0],
        ),
        HistogramSeries(
            name="a",
            color=QColor("#d85adf"),
            values=np.histogram(a_values, bins=256, range=(-128.0, 128.0))[0],
        ),
        HistogramSeries(
            name="b",
            color=QColor("#f0c646"),
            values=np.histogram(b_values, bins=256, range=(-128.0, 128.0))[0],
        ),
    )
    return HistogramPlotData(
        series=series,
        x_ticks=((0, "0"), (128, "128"), (255, "255")),
        x_label="Шкала: L 0..100; a,b -128..127",
        sample_count=int(rgb_pixels.shape[0]),
    )


def build_hsv_histogram(rgb_pixels: np.ndarray) -> HistogramPlotData | None:
    if rgb_pixels.size == 0:
        return None

    hsv_pixels = cv2.cvtColor(rgb_pixels.reshape((-1, 1, 3)), cv2.COLOR_RGB2HSV).reshape((-1, 3))
    hue = np.rint(hsv_pixels[:, 0].astype(np.float32) * (255.0 / 179.0))
    hue = np.clip(hue, 0, 255).astype(np.uint8)

    series = (
        HistogramSeries(
            name="H",
            color=QColor("#ff8a3d"),
            values=np.bincount(hue, minlength=256)[:256],
        ),
        HistogramSeries(
            name="S",
            color=QColor("#c46bff"),
            values=np.bincount(hsv_pixels[:, 1], minlength=256)[:256],
        ),
        HistogramSeries(
            name="V",
            color=QColor("#f1d35c"),
            values=np.bincount(hsv_pixels[:, 2], minlength=256)[:256],
        ),
    )
    return HistogramPlotData(
        series=series,
        x_ticks=((0, "0"), (64, "64"), (128, "128"), (192, "192"), (255, "255")),
        x_label="Шкала: H 0..360; S,V 0..255",
        sample_count=int(rgb_pixels.shape[0]),
    )


def build_yuv_histogram(rgb_pixels: np.ndarray) -> HistogramPlotData | None:
    if rgb_pixels.size == 0:
        return None

    yuv_pixels = cv2.cvtColor(rgb_pixels.reshape((-1, 1, 3)), cv2.COLOR_RGB2YUV).reshape((-1, 3))
    series = (
        HistogramSeries(
            name="Y",
            color=QColor("#eef2f7"),
            values=np.bincount(yuv_pixels[:, 0], minlength=256)[:256],
        ),
        HistogramSeries(
            name="U",
            color=QColor("#35d0ff"),
            values=np.bincount(yuv_pixels[:, 1], minlength=256)[:256],
        ),
        HistogramSeries(
            name="V",
            color=QColor("#ff6c91"),
            values=np.bincount(yuv_pixels[:, 2], minlength=256)[:256],
        ),
    )
    return HistogramPlotData(
        series=series,
        x_ticks=((0, "0"), (64, "64"), (128, "128"), (192, "192"), (255, "255")),
        x_label="Y яркость; U,V цветность, нейтраль около 128",
        sample_count=int(rgb_pixels.shape[0]),
    )


def build_lms_histogram(rgb_pixels: np.ndarray) -> HistogramPlotData | None:
    if rgb_pixels.size == 0:
        return None

    lms_pixels = _rgb_to_lms(rgb_pixels)
    lms_normalized = _normalize_lms_for_display(lms_pixels)

    series = (
        HistogramSeries(
            name="L",
            color=QColor("#ff5b63"),
            values=np.bincount(lms_normalized[:, 0], minlength=256)[:256],
        ),
        HistogramSeries(
            name="M",
            color=QColor("#35c46f"),
            values=np.bincount(lms_normalized[:, 1], minlength=256)[:256],
        ),
        HistogramSeries(
            name="S",
            color=QColor("#4aa3ff"),
            values=np.bincount(lms_normalized[:, 2], minlength=256)[:256],
        ),
    )
    return HistogramPlotData(
        series=series,
        x_ticks=((0, "0"), (64, "64"), (128, "128"), (192, "192"), (255, "255")),
        x_label="LMS нормирован к 0..255 для отображения",
        sample_count=int(rgb_pixels.shape[0]),
    )


def _rgb_to_lms(rgb_pixels: np.ndarray) -> np.ndarray:
    rgb = rgb_pixels.astype(np.float32) / 255.0
    linear_rgb = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    rgb_to_xyz_d65 = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz_to_lms_bradford = np.array(
        [
            [0.8951000, 0.2664000, -0.1614000],
            [-0.7502000, 1.7135000, 0.0367000],
            [0.0389000, -0.0685000, 1.0296000],
        ],
        dtype=np.float32,
    )

    # Linear sRGB -> XYZ D65 -> LMS using the Bradford cone-response matrix.
    xyz = linear_rgb @ rgb_to_xyz_d65.T
    lms = xyz @ xyz_to_lms_bradford.T
    return np.clip(lms, 0.0, None)


def _normalize_lms_for_display(lms_pixels: np.ndarray) -> np.ndarray:
    max_values = np.maximum(lms_pixels.max(axis=0), 1e-9)
    normalized = np.rint((lms_pixels / max_values) * 255.0)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _x_for_bin(plot_rect: QRectF, bin_index: int) -> float:
    bounded_index = max(0, min(255, int(bin_index)))
    return plot_rect.left() + (bounded_index / 255.0) * plot_rect.width()


def _font(point_size: int, *, bold: bool = False) -> QFont:
    font = QFont()
    font.setPointSize(point_size)
    font.setBold(bold)
    return font


def _short_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1_000:.0f}K"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _ensure_png_suffix(path: Path) -> Path:
    if path.suffix.lower() == ".png":
        return path
    if path.suffix:
        return path.with_suffix(".png")
    return path.with_suffix(".png")
