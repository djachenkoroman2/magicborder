from __future__ import annotations

import math
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import cv2
from PyQt5.QtCore import QDateTime, QSignalBlocker, Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QToolButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .canvas import ImageCanvas
from .detector import detect_leaf_contour
from .histograms import (
    HistogramPanel,
    build_hsv_histogram,
    build_lab_histogram,
    build_lms_histogram,
    build_rgb_histogram,
    build_yuv_histogram,
)
from .icons import ACTION_VISUALS, TOOLBAR_ICON_SIZE, apply_action_visual, load_icon
from .io_utils import (
    image_open_filter,
    load_annotation,
    load_project,
    load_raster_image,
    read_image_captured_at,
    save_annotation,
    save_project,
    SUPPORTED_RASTER_SUFFIXES,
    write_xlsx_table,
)
from .models import Annotation, Point, ProjectDocument, ProjectImageRecord, default_project_image_metadata
from .path_utils import portable_path_reference
from .property_browser import PropertyBrowser

APP_TITLE = "MagicBorder"
WORKSPACE_DEFAULT_SIZES = [260, 860, 280]
PROJECT_PANEL_DEFAULT_SIZES = [380, 220, 320]
HISTOGRAM_DEFAULT_SIZES = [170, 170, 170, 170, 170]
PROJECT_EXPORT_COLUMNS = [
    ("id", "ID изображения"),
    ("file_name", "Имя файла"),
    ("relative_path", "Относительный путь"),
    ("has_annotation", "Есть аннотация"),
    ("status", "Статус"),
    ("diagnosis", "Диагноз"),
    ("r", "Средний R"),
    ("g", "Средний G"),
    ("b", "Средний B"),
    ("contour_pixel_count", "Количество пикселов контура"),
]
PROJECT_EXPORT_FIELDNAMES = [field_name for field_name, _label in PROJECT_EXPORT_COLUMNS]
PROJECT_EXPORT_COLUMN_LABELS = dict(PROJECT_EXPORT_COLUMNS)
IMAGE_PROPERTY_GROUPS = [
    (
        "Общая информация о файле",
        [
            ("id", "ID"),
            ("file_name", "Файл"),
            ("relative_path", "Путь"),
            ("size", "Размер"),
            ("status", "Статус"),
            ("added_at", "Дата добавления"),
            ("captured_at", "Дата съёмки"),
        ],
    ),
    (
        "Информация о контуре",
        [
            ("annotation", "Аннотация"),
            ("point_count", "Количество узлов контура"),
            ("contour_pixel_count", "Количество пикселов контура"),
        ],
    ),
    (
        "Цветовое пространство RGB",
        [
            ("red", "Красный"),
            ("green", "Зелёный"),
            ("blue", "Синий"),
            ("average_color", "Средний цвет"),
        ],
    ),
    (
        "Локация",
        [
            ("illumination", "Освещённость"),
            ("humidity", "Влажность, %"),
            ("wind_speed", "Скорость ветра"),
            ("wind_direction", "Направление ветра"),
            ("latitude", "Широта"),
            ("longitude", "Долгота"),
        ],
    ),
    (
        "Дополнительно",
        [
            ("diagnosis", "Диагноз"),
            ("notes", "Дополнительные сведения"),
        ],
    ),
]
IMAGE_PROPERTY_EXPORT_ITEMS = [
    item
    for _group_title, group_items in IMAGE_PROPERTY_GROUPS
    for item in group_items
]
IMAGE_PROPERTY_EXPORT_KEYS = [field_name for field_name, _label in IMAGE_PROPERTY_EXPORT_ITEMS]
IMAGE_PROPERTY_EXPORT_LABELS = dict(IMAGE_PROPERTY_EXPORT_ITEMS)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 920)

        self.canvas = ImageCanvas(self)
        self.rgb_histogram_panel = HistogramPanel(
            "RGB-гистограмма",
            "rgb_histogram.png",
            self,
            default_file_name_provider=lambda: self._default_histogram_file_name("rgb"),
        )
        self.lab_histogram_panel = HistogramPanel(
            "Lab-гистограмма",
            "lab_histogram.png",
            self,
            default_file_name_provider=lambda: self._default_histogram_file_name("lab"),
        )
        self.hsv_histogram_panel = HistogramPanel(
            "HSV-гистограмма",
            "hsv_histogram.png",
            self,
            default_file_name_provider=lambda: self._default_histogram_file_name("hsv"),
        )
        self.yuv_histogram_panel = HistogramPanel(
            "YUV-гистограмма",
            "yuv_histogram.png",
            self,
            default_file_name_provider=lambda: self._default_histogram_file_name("yuv"),
        )
        self.lms_histogram_panel = HistogramPanel(
            "LMS-гистограмма",
            "lms_histogram.png",
            self,
            default_file_name_provider=lambda: self._default_histogram_file_name("lms"),
        )
        self._histogram_refresh_timer = QTimer(self)
        self._histogram_refresh_timer.setSingleShot(True)
        self._histogram_refresh_timer.timeout.connect(self._refresh_histograms)

        self.project_document: ProjectDocument | None = None
        self.project_path: Path | None = None
        self._current_project_image_id: str | None = None
        self._loading_project_image = False
        self._updating_project_list = False
        self._updating_project_info_fields = False
        self._updating_metadata_fields = False
        self._project_autosave_timer = QTimer(self)
        self._project_autosave_timer.setSingleShot(True)
        self._project_autosave_timer.timeout.connect(self._save_project_silently)

        self.setCentralWidget(self._create_workspace())

        self._current_annotation_path: Path | None = None

        self._create_actions()
        self._apply_action_visuals()
        self._create_menus()
        self._create_toolbar()

        self.statusBar().showMessage("Откройте фотографию листа растения.")
        self.canvas.message_changed.connect(self.statusBar().showMessage)
        self.canvas.image_state_changed.connect(self._update_action_states)
        self.canvas.image_state_changed.connect(self._schedule_histogram_refresh)
        self.canvas.image_state_changed.connect(self._update_project_properties)
        self.canvas.contour_state_changed.connect(self._update_action_states)
        self.canvas.contour_geometry_changed.connect(self._schedule_histogram_refresh)
        self.canvas.contour_geometry_changed.connect(self._handle_contour_geometry_changed)
        self._update_action_states()
        self._refresh_histograms()
        self._update_project_panel()

    def _create_workspace(self) -> QSplitter:
        project_panel = self._create_project_panel()
        analysis_panel = self._create_analysis_panel()

        self.workspace_splitter = QSplitter(Qt.Horizontal, self)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.addWidget(project_panel)
        self.workspace_splitter.addWidget(self.canvas)
        self.workspace_splitter.addWidget(analysis_panel)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 4)
        self.workspace_splitter.setStretchFactor(2, 1)
        self.workspace_splitter.setSizes(WORKSPACE_DEFAULT_SIZES)
        self.workspace_splitter.setStyleSheet(
            "QSplitter::handle { background: #d5dbe5; width: 8px; }"
            "QSplitter::handle:hover { background: #bdc7d5; }"
        )
        return self.workspace_splitter

    def _create_project_panel(self) -> QWidget:
        project_panel = QWidget(self)
        project_panel.setObjectName("projectPanel")
        project_panel.setMinimumWidth(220)

        self.project_list = QListWidget(project_panel)
        self.project_list.setObjectName("projectImageList")
        self.project_list.setAlternatingRowColors(True)
        self.project_list.currentItemChanged.connect(self._handle_project_selection_changed)

        images_title = QLabel("Изображения проекта")
        images_title.setObjectName("projectPanelTitle")

        self.export_project_excel_button = QToolButton(project_panel)
        self.export_project_excel_button.setText("Excel")
        self.export_project_excel_button.setIcon(load_icon("export-csv"))
        self.export_project_excel_button.setToolTip("Экспорт списка в Excel")
        self.export_project_excel_button.setStatusTip(
            "Экспорт списка в Excel: сохранить список изображений проекта в файл .xlsx."
        )
        self.export_project_excel_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.export_project_excel_button.clicked.connect(self.export_project_excel)
        self.export_project_csv_button = self.export_project_excel_button

        images_header_layout = QHBoxLayout()
        images_header_layout.setContentsMargins(0, 0, 0, 0)
        images_header_layout.setSpacing(6)
        images_header_layout.addWidget(images_title)
        images_header_layout.addStretch(1)
        images_header_layout.addWidget(self.export_project_excel_button)

        images_widget = QWidget(project_panel)
        images_widget.setObjectName("projectImagesPanel")
        images_layout = QVBoxLayout(images_widget)
        images_layout.setContentsMargins(8, 8, 8, 8)
        images_layout.setSpacing(6)
        images_layout.addLayout(images_header_layout)
        images_layout.addWidget(self.project_list, 1)

        project_properties_widget = self._create_project_properties_panel(project_panel)
        properties_widget = self._create_image_properties_panel(project_panel)

        self.project_splitter = QSplitter(Qt.Vertical, project_panel)
        self.project_splitter.setChildrenCollapsible(False)
        self.project_splitter.addWidget(images_widget)
        self.project_splitter.addWidget(project_properties_widget)
        self.project_splitter.addWidget(properties_widget)
        self.project_splitter.setSizes(PROJECT_PANEL_DEFAULT_SIZES)
        self.project_splitter.setStyleSheet(
            "QSplitter::handle { background: #d5dbe5; height: 7px; }"
            "QSplitter::handle:hover { background: #bdc7d5; }"
        )

        layout = QVBoxLayout(project_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.project_splitter)

        project_panel.setStyleSheet(
            "QWidget#projectPanel { background: #f5f7fb; border-right: 1px solid #d6dde8; }"
            "QWidget#projectImagesPanel { background: #f7fbff; }"
            "QWidget#projectPropertiesPanel { background: #f8fcf6; }"
            "QWidget#imagePropertiesPanel { background: #fffaf4; }"
            "QLabel#projectPanelTitle { color: #1f2937; font-size: 13px; font-weight: 600; }"
            "QListWidget#projectImageList { background: #ffffff; color: #1f2937; border: 1px solid #ccd6e1; border-radius: 5px; outline: none; }"
            "QListWidget#projectImageList::item { padding: 7px 8px; border-bottom: 1px solid #e5eaf2; }"
            "QListWidget#projectImageList::item:selected { background: #d8edf3; color: #102a32; }"
            "QListWidget#projectImageList::item:alternate { background: #f7f9fc; }"
            "QTreeWidget#propertyBrowser { background: transparent; border: none; outline: none; color: #1f2937; }"
            "QTreeWidget#propertyBrowser::item { padding: 2px 0; }"
            "QTreeWidget#propertyBrowser::item:hover { color: #0f766e; }"
            "QLabel#propertyValue { color: #18202c; }"
            "QLabel#propertyEmpty { color: #7a8696; padding: 12px; }"
            "QLineEdit, QTextEdit { background: #ffffff; color: #1f2937; border: 1px solid #ccd6e1; border-radius: 4px; padding: 4px; selection-background-color: #cdeaf2; selection-color: #102a32; }"
            "QLineEdit:focus, QTextEdit:focus { border-color: #2A9D8F; }"
            "QLineEdit:read-only { color: #5f6b7a; background: #f1f4f8; }"
            "QFrame#averageColorSwatch { border: 1px solid #95a3b8; border-radius: 4px; background: transparent; }"
            "QToolButton { border: 1px solid transparent; border-radius: 4px; padding: 4px; color: #1f2937; }"
            "QToolButton:hover { background: #e9eef5; border-color: #c6d1de; }"
            "QToolButton:pressed { background: #dce4ee; }"
        )
        return project_panel

    def _create_project_properties_panel(self, parent: QWidget) -> QWidget:
        properties_widget = QWidget(parent)
        properties_widget.setObjectName("projectPropertiesPanel")
        self.project_properties_panel = properties_widget

        title = QLabel("Свойства проекта")
        title.setObjectName("projectPanelTitle")

        self.project_properties_empty_label = QLabel("Проект не открыт.")
        self.project_properties_empty_label.setObjectName("propertyEmpty")
        self.project_properties_empty_label.setWordWrap(True)
        self.project_properties_empty_label.setAlignment(Qt.AlignCenter)

        self.project_general_info = QTextEdit(properties_widget)
        self.project_general_info.setAcceptRichText(False)
        self.project_general_info.setMinimumHeight(58)
        self.project_general_info.textChanged.connect(self._handle_project_general_info_changed)
        self.project_image_count = self._property_value_label()
        self.project_mean_red = self._property_value_label()
        self.project_mean_green = self._property_value_label()
        self.project_mean_blue = self._property_value_label()

        self.project_properties_browser = PropertyBrowser(properties_widget)
        self._add_property_browser_group(
            self.project_properties_browser,
            "Общие свойства",
            [
                ("Общая информация", self.project_general_info),
                ("Количество изображений", self.project_image_count),
            ],
        )
        self._add_property_browser_group(
            self.project_properties_browser,
            "Цветовое пространство RGB",
            [
                ("Средний R", self.project_mean_red),
                ("Средний G", self.project_mean_green),
                ("Средний B", self.project_mean_blue),
            ],
        )

        layout = QVBoxLayout(properties_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(self.project_properties_empty_label, 1)
        layout.addWidget(self.project_properties_browser, 1)
        return properties_widget

    def _create_image_properties_panel(self, parent: QWidget) -> QWidget:
        properties_widget = QWidget(parent)
        properties_widget.setObjectName("imagePropertiesPanel")
        self.image_properties_panel = properties_widget

        title = QLabel("Свойства изображения")
        title.setObjectName("projectPanelTitle")

        self.export_image_properties_excel_button = QToolButton(properties_widget)
        self.export_image_properties_excel_button.setText("Excel")
        self.export_image_properties_excel_button.setIcon(load_icon("export-csv"))
        self.export_image_properties_excel_button.setToolTip("Экспорт свойств изображения в Excel")
        self.export_image_properties_excel_button.setStatusTip(
            "Сохранить свойства выбранного изображения в файл .xlsx."
        )
        self.export_image_properties_excel_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.export_image_properties_excel_button.clicked.connect(self.export_image_properties_excel)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.export_image_properties_excel_button)

        self.properties_empty_label = QLabel("Изображение не выбрано.")
        self.properties_empty_label.setObjectName("propertyEmpty")
        self.properties_empty_label.setWordWrap(True)
        self.properties_empty_label.setAlignment(Qt.AlignCenter)

        self.property_file_name = self._metadata_line_edit()
        self.property_file_name.editingFinished.connect(self._handle_file_name_edit_finished)
        self.property_file_name_widget, self.rename_file_as_id_button = self._file_name_widget(properties_widget)
        self.property_path = self._property_value_label()
        self.property_size = self._property_value_label()
        self.property_annotation = self._property_value_label()
        self.property_points = self._property_value_label()
        self.property_contour_pixels = self._property_value_label()
        self.property_red = self._property_value_label()
        self.property_green = self._property_value_label()
        self.property_blue = self._property_value_label()
        self.property_status = self._property_value_label()
        self.average_color_swatch = QFrame(properties_widget)
        self.average_color_swatch.setObjectName("averageColorSwatch")
        self.average_color_swatch.setFixedSize(58, 24)

        self.image_id = self._metadata_line_edit()
        self.image_id.editingFinished.connect(self._handle_image_id_edit_finished)
        self.image_id_widget, self.generate_image_id_button = self._image_id_widget(
            self.image_id,
            properties_widget,
        )
        self.metadata_added_at = self._metadata_line_edit()
        self.metadata_captured_at = self._metadata_line_edit()
        self.metadata_added_at_widget, self.metadata_added_at_button = self._metadata_datetime_widget(
            self.metadata_added_at,
            "added_at",
            properties_widget,
        )
        self.metadata_captured_at_widget, self.metadata_captured_at_button = self._metadata_datetime_widget(
            self.metadata_captured_at,
            "captured_at",
            properties_widget,
        )
        self.metadata_illumination = self._metadata_line_edit()
        self.metadata_humidity = self._metadata_line_edit()
        self.metadata_wind_speed = self._metadata_line_edit()
        self.metadata_wind_direction = self._metadata_line_edit()
        self.metadata_latitude = self._metadata_line_edit()
        self.metadata_longitude = self._metadata_line_edit()
        self.metadata_diagnosis = self._metadata_line_edit()
        self.metadata_notes = QTextEdit(properties_widget)
        self.metadata_notes.setAcceptRichText(False)
        self.metadata_notes.setMinimumHeight(70)
        self.metadata_notes.textChanged.connect(self._handle_metadata_notes_changed)

        self._metadata_fields = {
            "added_at": self.metadata_added_at,
            "captured_at": self.metadata_captured_at,
            "illumination": self.metadata_illumination,
            "humidity": self.metadata_humidity,
            "wind_speed": self.metadata_wind_speed,
            "wind_direction": self.metadata_wind_direction,
            "latitude": self.metadata_latitude,
            "longitude": self.metadata_longitude,
            "diagnosis": self.metadata_diagnosis,
        }
        for metadata_key, field in self._metadata_fields.items():
            field.editingFinished.connect(
                lambda key=metadata_key, editor=field: self._handle_metadata_line_edit_finished(key, editor)
            )

        self.properties_browser = PropertyBrowser(properties_widget)
        self._add_property_browser_group(
            self.properties_browser,
            "Общая информация о файле",
            [
                ("ID", self.image_id_widget),
                ("Файл", self.property_file_name_widget),
                ("Путь", self.property_path),
                ("Размер", self.property_size),
                ("Статус", self.property_status),
                ("Дата добавления", self.metadata_added_at_widget),
                ("Дата съёмки", self.metadata_captured_at_widget),
            ],
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Информация о контуре",
            [
                ("Аннотация", self.property_annotation),
                ("Количество узлов контура", self.property_points),
                ("Количество пикселов контура", self.property_contour_pixels),
            ],
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Цветовое пространство RGB",
            [
                ("Красный", self.property_red),
                ("Зелёный", self.property_green),
                ("Синий", self.property_blue),
                ("Средний цвет", self.average_color_swatch),
            ],
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Локация",
            [
                ("Освещённость", self.metadata_illumination),
                ("Влажность, %", self.metadata_humidity),
                ("Скорость ветра", self.metadata_wind_speed),
                ("Направление ветра", self.metadata_wind_direction),
                ("Широта", self.metadata_latitude),
                ("Долгота", self.metadata_longitude),
            ],
        )
        self._add_property_browser_group(
            self.properties_browser,
            "Дополнительно",
            [
                ("Диагноз", self.metadata_diagnosis),
                ("Дополнительные сведения", self.metadata_notes),
            ],
        )

        layout = QVBoxLayout(properties_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(header_layout)
        layout.addWidget(self.properties_empty_label, 1)
        layout.addWidget(self.properties_browser, 1)
        return properties_widget

    def _add_property_browser_group(
        self,
        browser: PropertyBrowser,
        title: str,
        rows: list[tuple[str, QWidget]],
    ) -> None:
        browser.add_group(title, expanded=False)
        for label, widget in rows:
            browser.add_property(title, label, widget)

    def _property_value_label(self) -> QLabel:
        label = QLabel("-")
        label.setObjectName("propertyValue")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label

    def _metadata_line_edit(self) -> QLineEdit:
        field = QLineEdit()
        field.setClearButtonEnabled(True)
        return field

    def _file_name_widget(self, parent: QWidget) -> tuple[QWidget, QToolButton]:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        button = QToolButton(container)
        button.setText("ID")
        button.setToolTip("Переименовать файл как ID")
        button.setStatusTip("Переименовать файл изображения по текущему ID с сохранением расширения.")
        button.setAutoRaise(True)
        button.clicked.connect(self._rename_file_to_image_id)

        layout.addWidget(self.property_file_name, 1)
        layout.addWidget(button)
        return container, button

    def _image_id_widget(self, field: QLineEdit, parent: QWidget) -> tuple[QWidget, QToolButton]:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        button = QToolButton(container)
        button.setIcon(load_icon("generate-id"))
        button.setToolTip("Сгенерировать ID")
        button.setStatusTip("Сгенерировать новый GUID изображения.")
        button.setAutoRaise(True)
        button.clicked.connect(self._generate_image_id)

        layout.addWidget(field, 1)
        layout.addWidget(button)
        return container, button

    def _metadata_datetime_widget(
        self,
        field: QLineEdit,
        metadata_key: str,
        parent: QWidget,
    ) -> tuple[QWidget, QToolButton]:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        button = QToolButton(container)
        button.setIcon(load_icon("calendar-time"))
        button.setToolTip("Выбрать дату и время")
        button.setStatusTip("Открыть диалог выбора даты и времени.")
        button.setAutoRaise(True)
        button.clicked.connect(lambda: self._open_metadata_datetime_dialog(metadata_key, field))

        layout.addWidget(field, 1)
        layout.addWidget(button)
        return container, button

    def _create_analysis_panel(self) -> QWidget:
        analysis_panel = QWidget(self)
        analysis_panel.setObjectName("analysisPanel")

        analysis_title = QLabel("Свойства и аналитика")
        analysis_title.setObjectName("analysisTitle")

        self.histogram_splitter = QSplitter(Qt.Vertical, analysis_panel)
        self.histogram_splitter.setChildrenCollapsible(False)
        self.histogram_splitter.addWidget(self.rgb_histogram_panel)
        self.histogram_splitter.addWidget(self.lab_histogram_panel)
        self.histogram_splitter.addWidget(self.hsv_histogram_panel)
        self.histogram_splitter.addWidget(self.yuv_histogram_panel)
        self.histogram_splitter.addWidget(self.lms_histogram_panel)
        self.histogram_splitter.setSizes(HISTOGRAM_DEFAULT_SIZES)
        self.histogram_splitter.setStyleSheet(
            "QSplitter::handle { background: #d5dbe5; height: 7px; }"
            "QSplitter::handle:hover { background: #bdc7d5; }"
        )

        analysis_layout = QVBoxLayout(analysis_panel)
        analysis_layout.setContentsMargins(10, 10, 10, 10)
        analysis_layout.setSpacing(8)
        analysis_layout.addWidget(analysis_title)
        analysis_layout.addWidget(self.histogram_splitter, 1)

        analysis_panel.setStyleSheet(
            "QWidget#analysisPanel { background: #f7f9fc; border-left: 1px solid #d6dde8; }"
            "QLabel#analysisTitle { color: #1f2937; font-size: 13px; font-weight: 600; }"
            "QFrame#histogramPanel { background: #ffffff; border: 1px solid #ccd6e1; border-radius: 6px; }"
            "QLabel#histogramTitle { color: #1f2937; font-size: 12px; font-weight: 600; }"
            "QToolButton { border: 1px solid transparent; border-radius: 4px; padding: 4px; }"
            "QToolButton:hover { background: #e9eef5; border-color: #c6d1de; }"
            "QToolButton:pressed { background: #dce4ee; }"
        )
        return analysis_panel

    def _create_actions(self) -> None:
        self.new_project_action = QAction("Новый проект...", self)
        self.new_project_action.setShortcut("Ctrl+N")
        self.new_project_action.triggered.connect(self.new_project)

        self.open_project_action = QAction("Открыть проект...", self)
        self.open_project_action.setShortcut("Ctrl+Alt+O")
        self.open_project_action.triggered.connect(self.open_project)

        self.save_project_action = QAction("Сохранить проект", self)
        self.save_project_action.setShortcut("Ctrl+Shift+S")
        self.save_project_action.triggered.connect(self.save_project_file)

        self.close_project_action = QAction("Закрыть проект", self)
        self.close_project_action.setShortcut("Ctrl+W")
        self.close_project_action.triggered.connect(self.close_project)

        self.add_images_action = QAction("Добавить изображения...", self)
        self.add_images_action.setShortcut("Ctrl+I")
        self.add_images_action.triggered.connect(self.add_images_to_project)

        self.sync_images_action = QAction("Синхронизировать папку изображений", self)
        self.sync_images_action.triggered.connect(self.sync_project_images_folder)

        self.remove_image_action = QAction("Удалить изображение из проекта", self)
        self.remove_image_action.triggered.connect(self.remove_selected_project_image)

        self.export_project_excel_action = QAction("Экспорт списка в Excel...", self)
        self.export_project_excel_action.triggered.connect(self.export_project_excel)
        self.export_project_csv_action = self.export_project_excel_action

        self.export_image_properties_excel_action = QAction("Экспорт свойств изображения в Excel...", self)
        self.export_image_properties_excel_action.triggered.connect(self.export_image_properties_excel)

        self.exit_action = QAction("Выход", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)

        self.zoom_in_action = QAction("Увеличить", self)
        self.zoom_in_action.setShortcut("Ctrl++")
        self.zoom_in_action.triggered.connect(self.canvas.zoom_in)

        self.zoom_out_action = QAction("Уменьшить", self)
        self.zoom_out_action.setShortcut("Ctrl+-")
        self.zoom_out_action.triggered.connect(self.canvas.zoom_out)

        self.fit_image_action = QAction("Показать целиком", self)
        self.fit_image_action.setShortcut("Ctrl+9")
        self.fit_image_action.triggered.connect(self.canvas.fit_to_image)

        self.actual_size_action = QAction("100%", self)
        self.actual_size_action.setShortcut("Ctrl+0")
        self.actual_size_action.triggered.connect(self.canvas.reset_zoom)

        self.default_view_action = QAction("Вид по умолчанию", self)
        self.default_view_action.setShortcut("Ctrl+R")
        self.default_view_action.triggered.connect(self.restore_default_view)

        self.new_contour_action = QAction("Новый контур", self)
        self.new_contour_action.setShortcut("Ctrl+Shift+N")
        self.new_contour_action.triggered.connect(self.create_new_contour)

        self.detect_contour_action = QAction("Определить контур", self)
        self.detect_contour_action.setShortcut("F5")
        self.detect_contour_action.triggered.connect(self.detect_contour)

        self.delete_contour_action = QAction("Удалить контур", self)
        self.delete_contour_action.setShortcut("Ctrl+Delete")
        self.delete_contour_action.triggered.connect(self.delete_current_contour)

        self.flatten_background_action = QAction("Выровнять фон", self)
        self.flatten_background_action.setShortcut("F6")
        self.flatten_background_action.triggered.connect(self.flatten_background)

        self.save_annotation_action = QAction("Сохранить аннотацию...", self)
        self.save_annotation_action.setShortcut("Ctrl+Alt+S")
        self.save_annotation_action.triggered.connect(self.save_annotation_file)

        self.open_annotation_action = QAction("Открыть аннотацию...", self)
        self.open_annotation_action.setShortcut("Ctrl+Shift+O")
        self.open_annotation_action.triggered.connect(self.open_annotation_file)

        self.about_action = QAction("О программе", self)
        self.about_action.triggered.connect(self.show_about_dialog)

    def _apply_action_visuals(self) -> None:
        action_map = {
            "new_project": self.new_project_action,
            "open_project": self.open_project_action,
            "save_project": self.save_project_action,
            "close_project": self.close_project_action,
            "add_images": self.add_images_action,
            "sync_images": self.sync_images_action,
            "remove_image": self.remove_image_action,
            "export_project_excel": self.export_project_excel_action,
            "export_image_properties_excel": self.export_image_properties_excel_action,
            "exit": self.exit_action,
            "zoom_in": self.zoom_in_action,
            "zoom_out": self.zoom_out_action,
            "fit_image": self.fit_image_action,
            "actual_size": self.actual_size_action,
            "default_view": self.default_view_action,
            "new_contour": self.new_contour_action,
            "detect_contour": self.detect_contour_action,
            "delete_contour": self.delete_contour_action,
            "flatten_background": self.flatten_background_action,
            "save_annotation": self.save_annotation_action,
            "open_annotation": self.open_annotation_action,
            "about": self.about_action,
        }

        for action_name, action in action_map.items():
            apply_action_visual(action, ACTION_VISUALS[action_name])

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("Файл")
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.open_project_action)
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.close_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.add_images_action)
        file_menu.addAction(self.sync_images_action)
        file_menu.addAction(self.remove_image_action)
        file_menu.addAction(self.export_project_excel_action)
        file_menu.addAction(self.export_image_properties_excel_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        view_menu = self.menuBar().addMenu("Вид")
        view_menu.addAction(self.zoom_in_action)
        view_menu.addAction(self.zoom_out_action)
        view_menu.addAction(self.fit_image_action)
        view_menu.addAction(self.actual_size_action)
        view_menu.addSeparator()
        view_menu.addAction(self.default_view_action)

        tools_menu = self.menuBar().addMenu("Инструменты")
        tools_menu.addAction(self.new_contour_action)
        tools_menu.addAction(self.detect_contour_action)
        tools_menu.addAction(self.delete_contour_action)
        tools_menu.addAction(self.flatten_background_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.save_annotation_action)
        tools_menu.addAction(self.open_annotation_action)

        help_menu = self.menuBar().addMenu("Помощь")
        help_menu.addAction(self.about_action)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Основная панель", self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        toolbar.setIconSize(TOOLBAR_ICON_SIZE)
        toolbar.setStyleSheet(
            "QToolBar { spacing: 4px; }"
            "QToolButton { padding: 6px; min-width: 38px; min-height: 38px; }"
        )
        self.addToolBar(toolbar)

        toolbar.addAction(self.new_project_action)
        toolbar.addAction(self.open_project_action)
        toolbar.addAction(self.save_project_action)
        toolbar.addAction(self.close_project_action)
        toolbar.addSeparator()
        toolbar.addAction(self.add_images_action)
        toolbar.addAction(self.sync_images_action)
        toolbar.addAction(self.remove_image_action)
        toolbar.addAction(self.export_project_excel_action)
        toolbar.addAction(self.export_image_properties_excel_action)
        toolbar.addSeparator()
        toolbar.addAction(self.zoom_in_action)
        toolbar.addAction(self.zoom_out_action)
        toolbar.addAction(self.fit_image_action)
        toolbar.addAction(self.actual_size_action)
        toolbar.addAction(self.default_view_action)
        toolbar.addSeparator()
        toolbar.addAction(self.new_contour_action)
        toolbar.addAction(self.detect_contour_action)
        toolbar.addAction(self.delete_contour_action)
        toolbar.addAction(self.flatten_background_action)
        toolbar.addAction(self.save_annotation_action)
        toolbar.addAction(self.open_annotation_action)
        toolbar.addSeparator()
        toolbar.addAction(self.about_action)
        toolbar_spacer = QWidget(toolbar)
        toolbar_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(toolbar_spacer)
        toolbar.addAction(self.exit_action)

    def _update_action_states(self, *_args) -> None:
        has_image = self.canvas.has_image()
        has_contour = self.canvas.has_contour()
        has_project = self.project_document is not None
        selected_project_image = self._selected_project_image()
        has_project_image = has_project and selected_project_image is not None
        has_project_contour = (
            selected_project_image is not None
            and (
                selected_project_image.annotation is not None
                or selected_project_image.raw_annotation is not None
            )
        )

        self.save_project_action.setEnabled(has_project)
        self.close_project_action.setEnabled(has_project)
        self.add_images_action.setEnabled(has_project)
        self.sync_images_action.setEnabled(has_project)
        self.remove_image_action.setEnabled(has_project_image)
        self.export_project_excel_action.setEnabled(has_project)
        self.export_project_excel_button.setEnabled(has_project)
        self.export_image_properties_excel_action.setEnabled(has_project_image)
        self.export_image_properties_excel_button.setEnabled(has_project_image)
        self.zoom_in_action.setEnabled(has_image)
        self.zoom_out_action.setEnabled(has_image)
        self.fit_image_action.setEnabled(has_image)
        self.actual_size_action.setEnabled(has_image)
        self.new_contour_action.setEnabled(has_project_image and has_image)
        self.detect_contour_action.setEnabled(has_project_image and has_image)
        self.delete_contour_action.setEnabled(has_project_image and (has_contour or has_project_contour))
        self.flatten_background_action.setEnabled(has_project_image and has_image and has_contour)
        self.save_annotation_action.setEnabled(has_project_image and has_contour)
        self.open_annotation_action.setEnabled(has_project_image)

    def restore_default_view(self) -> None:
        self._restore_splitter_defaults()
        if self.canvas.has_image():
            self.canvas.fit_to_image()
        self.statusBar().showMessage("Вид по умолчанию восстановлен.")

    def _schedule_histogram_refresh(self, *_args) -> None:
        if not self._histogram_refresh_timer.isActive():
            self._histogram_refresh_timer.start(90)

    def _refresh_histograms(self) -> None:
        try:
            rgb_pixels = self.canvas.contour_rgb_pixels()
        except ValueError as exc:
            self._clear_histograms(str(exc))
            return

        if rgb_pixels.size == 0:
            message = "Откройте изображение и создайте контур, чтобы увидеть гистограмму."
            if self.canvas.has_image() and not self.canvas.has_contour():
                message = "Создайте основной контур, чтобы увидеть гистограмму."
            elif self.canvas.has_image() and self.canvas.has_contour():
                message = "Внутри контура нет пикселей для анализа."
            self._clear_histograms(message)
            return

        histogram_specs = (
            (self.rgb_histogram_panel, build_rgb_histogram, "RGB"),
            (self.lab_histogram_panel, build_lab_histogram, "Lab"),
            (self.hsv_histogram_panel, build_hsv_histogram, "HSV"),
            (self.yuv_histogram_panel, build_yuv_histogram, "YUV"),
            (self.lms_histogram_panel, build_lms_histogram, "LMS"),
        )
        for panel, build_histogram, name in histogram_specs:
            histogram = build_histogram(rgb_pixels)
            if histogram is None:
                panel.clear_histogram(f"Внутри контура нет пикселей для {name}-гистограммы.")
            else:
                panel.set_histogram(histogram)

    def _clear_histograms(self, message: str) -> None:
        for panel in (
            self.rgb_histogram_panel,
            self.lab_histogram_panel,
            self.hsv_histogram_panel,
            self.yuv_histogram_panel,
            self.lms_histogram_panel,
        ):
            panel.clear_histogram(message)

    def _restore_splitter_defaults(self) -> None:
        self.workspace_splitter.setSizes(WORKSPACE_DEFAULT_SIZES)
        self.project_splitter.setSizes(PROJECT_PANEL_DEFAULT_SIZES)
        self.histogram_splitter.setSizes(HISTOGRAM_DEFAULT_SIZES)

    def new_project(self) -> None:
        project_name, accepted = QInputDialog.getText(
            self,
            "Новый проект",
            "Имя проекта:",
        )
        if not accepted:
            return

        safe_name = _safe_project_name(project_name)
        if not safe_name:
            self._show_warning("Некорректное имя", "Введите имя проекта.")
            return

        parent_dir_name = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку, где будет создан проект",
            "",
        )
        if not parent_dir_name:
            return

        project_dir = Path(parent_dir_name) / safe_name
        if project_dir.exists() and any(project_dir.iterdir()):
            self._show_warning(
                "Папка уже существует",
                "Выберите другое имя проекта или пустую папку.",
            )
            return

        self._save_current_project_annotation()
        if not self._save_project_silently(show_error=True):
            return

        try:
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / "images").mkdir(exist_ok=True)
        except OSError as exc:
            self._show_error("Ошибка создания проекта", f"Не удалось создать папку проекта: {exc}")
            return

        project_path = project_dir / f"{safe_name}.json"
        document = ProjectDocument(name=safe_name, images=[])
        try:
            save_project(project_path, document)
        except OSError as exc:
            self._show_error("Ошибка создания проекта", f"Не удалось сохранить файл проекта: {exc}")
            return

        self._set_project(project_path, document)
        self.statusBar().showMessage(f"Создан проект: {safe_name}")

    def open_project(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть проект",
            "",
            "MagicBorder project (*.json);;JSON (*.json);;All files (*)",
        )
        if not file_name:
            return

        self._save_current_project_annotation()
        if not self._save_project_silently(show_error=True):
            return

        project_path = Path(file_name)
        try:
            document = load_project(project_path)
        except (OSError, ValueError) as exc:
            self._show_error("Ошибка открытия проекта", str(exc))
            return

        self._set_project(project_path, document)
        self.statusBar().showMessage(f"Открыт проект: {document.name}")

    def save_project_file(self) -> None:
        if self.project_document is None:
            self._show_warning("Нет проекта", "Сначала создайте или откройте проект.")
            return
        self._save_current_project_annotation()
        if self._save_project_silently(show_error=True):
            self.statusBar().showMessage("Проект сохранён.")

    def close_project(self) -> None:
        if self.project_document is None:
            return
        self._save_current_project_annotation()
        if not self._save_project_silently(show_error=True):
            return
        self._clear_project_state()
        self.statusBar().showMessage("Проект закрыт.")

    def add_images_to_project(self) -> None:
        if self.project_document is None or self.project_path is None:
            self._show_warning("Нет проекта", "Сначала создайте или откройте проект.")
            return

        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Добавить изображения в проект",
            "",
            image_open_filter(),
        )
        if not file_names:
            return

        project_dir = self.project_path.parent
        image_dir = project_dir / self.project_document.images_dir
        errors: list[str] = []
        added_ids: list[str] = []

        try:
            image_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._show_error("Ошибка добавления", f"Не удалось создать папку изображений: {exc}")
            return

        for file_name in file_names:
            source_path = Path(file_name)
            try:
                loaded_image = load_raster_image(source_path)
                destination_path = _unique_destination_path(image_dir, source_path.name)
                if source_path.resolve() != destination_path.resolve():
                    shutil.copy2(source_path, destination_path)
            except (OSError, ValueError) as exc:
                errors.append(f"{source_path.name}: {exc}")
                continue

            record_id = self._new_project_image_id()
            record = ProjectImageRecord(
                id=record_id,
                relative_path=portable_path_reference(destination_path, project_dir),
                display_name=destination_path.name,
                image_width=loaded_image.width,
                image_height=loaded_image.height,
                metadata=default_project_image_metadata(
                    added_at=_current_timestamp(),
                    captured_at=read_image_captured_at(source_path),
                ),
            )
            self.project_document.images.append(record)
            added_ids.append(record.id)

        if added_ids:
            self._refresh_project_list()
            self._select_project_image(added_ids[0])
            self._update_project_summary_properties()
            self._save_project_silently(show_error=True)
            self.statusBar().showMessage(f"Добавлено изображений: {len(added_ids)}")

        if errors:
            self._show_warning("Не все изображения добавлены", "\n".join(errors[:8]))

    def sync_project_images_folder(self) -> None:
        if self.project_document is None or self.project_path is None:
            self._show_warning("Нет проекта", "Сначала создайте или откройте проект.")
            return

        self._save_current_project_annotation()

        project_dir = self.project_path.parent
        image_dir = project_dir / self.project_document.images_dir
        errors: list[str] = []
        added_ids: list[str] = []

        try:
            image_dir.mkdir(parents=True, exist_ok=True)
            candidate_paths = sorted(
                path
                for path in image_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_RASTER_SUFFIXES
            )
        except OSError as exc:
            self._show_error(
                "Ошибка синхронизации",
                f"Не удалось прочитать папку изображений проекта: {exc}",
            )
            return

        existing_paths = {
            _project_relative_path_key(record.relative_path)
            for record in self.project_document.images
        }

        for image_path in candidate_paths:
            relative_path = portable_path_reference(image_path, project_dir)
            relative_path_key = _project_relative_path_key(relative_path)
            if relative_path_key in existing_paths:
                continue

            try:
                loaded_image = load_raster_image(image_path)
            except (OSError, ValueError) as exc:
                errors.append(f"{relative_path}: {exc}")
                continue

            try:
                record_id = self._new_project_image_id()
            except ValueError as exc:
                self._show_error("Ошибка синхронизации", str(exc))
                return

            record = ProjectImageRecord(
                id=record_id,
                relative_path=relative_path,
                display_name=image_path.name,
                image_width=loaded_image.width,
                image_height=loaded_image.height,
                metadata=default_project_image_metadata(
                    added_at=_current_timestamp(),
                    captured_at=read_image_captured_at(image_path),
                ),
            )
            self.project_document.images.append(record)
            existing_paths.add(relative_path_key)
            added_ids.append(record.id)

        if added_ids:
            self._refresh_project_list()
            self._select_project_image(added_ids[0])
            self._update_project_summary_properties()
            self._save_project_silently(show_error=True)
            self.statusBar().showMessage(f"Синхронизировано изображений: {len(added_ids)}")
        else:
            self.statusBar().showMessage("Новых изображений не найдено.")

        if errors:
            self._show_warning("Не все изображения синхронизированы", "\n".join(errors[:8]))

    def remove_selected_project_image(self) -> None:
        if self.project_document is None or self.project_path is None:
            return

        record = self._selected_project_image()
        if record is None:
            self._show_warning("Изображение не выбрано", "Выберите изображение в списке проекта.")
            return

        row_to_remove = self.project_list.currentRow()
        self._save_current_project_annotation()

        dialog = QMessageBox(self)
        dialog.setWindowTitle("Удалить изображение")
        dialog.setIcon(QMessageBox.Question)
        dialog.setText(f"Удалить '{record.display_name}' из проекта?")
        delete_file_button = dialog.addButton("Удалить файл", QMessageBox.DestructiveRole)
        remove_only_button = dialog.addButton("Только убрать из проекта", QMessageBox.AcceptRole)
        cancel_button = dialog.addButton("Отмена", QMessageBox.RejectRole)
        dialog.setDefaultButton(remove_only_button)
        dialog.exec_()

        clicked_button = dialog.clickedButton()
        if clicked_button is cancel_button:
            return

        image_path = self._project_image_path(record)
        if clicked_button is delete_file_button and image_path.exists():
            try:
                image_path.unlink()
            except OSError as exc:
                self._show_error("Ошибка удаления", f"Не удалось удалить файл изображения: {exc}")
                return

        self.project_document.images = [
            item for item in self.project_document.images if item.id != record.id
        ]
        if self._current_project_image_id == record.id:
            self._current_project_image_id = None

        self._refresh_project_list()
        if self.project_document.images:
            next_row = min(max(row_to_remove, 0), len(self.project_document.images) - 1)
            self.project_list.setCurrentRow(next_row)
        else:
            self._clear_current_image_display()
        self._update_project_summary_properties()
        self._save_project_silently(show_error=True)
        self.statusBar().showMessage("Изображение удалено из проекта.")

    def export_project_excel(self) -> None:
        if self.project_document is None or self.project_path is None:
            self._show_warning("Нет проекта", "Сначала создайте или откройте проект.")
            return

        self._save_current_project_annotation()
        selected_fieldnames = self._select_project_export_columns()
        if selected_fieldnames is None:
            return
        if not selected_fieldnames:
            self._show_warning("Нет выбранных столбцов", "Выберите хотя бы один столбец для экспорта.")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт списка в Excel",
            str(self.project_path.with_suffix(".xlsx")),
            "Excel (*.xlsx);;All files (*)",
        )
        if not file_name:
            return

        output_path = _ensure_xlsx_suffix(Path(file_name))
        try:
            self._write_project_excel(output_path, selected_fieldnames)
        except OSError as exc:
            self._show_error("Ошибка экспорта списка в Excel", f"Не удалось сохранить Excel-файл: {exc}")
            return

        self.statusBar().showMessage(f"Экспорт списка в Excel выполнен: {output_path.name}")

    def export_project_csv(self) -> None:
        self.export_project_excel()

    def _select_project_export_columns(self) -> list[str] | None:
        return self._select_checked_export_items(
            title="Столбцы экспорта",
            items=PROJECT_EXPORT_COLUMNS,
            empty_title="Нет выбранных столбцов",
            empty_message="Выберите хотя бы один столбец для экспорта.",
        )

    def _select_checked_export_items(
        self,
        *,
        title: str,
        items: list[tuple[str, str]] | None = None,
        grouped_items: list[tuple[str, list[tuple[str, str]]]] | None = None,
        empty_title: str,
        empty_message: str,
    ) -> list[str] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)

        item_list = QListWidget(dialog)
        item_list.setAlternatingRowColors(True)

        def add_checkable_item(field_name: str, label: str) -> None:
            item = QListWidgetItem(label)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, field_name)
            item_list.addItem(item)

        if grouped_items is not None:
            for group_title, group_items in grouped_items:
                group_item = QListWidgetItem(group_title)
                group_item.setFlags(Qt.ItemIsEnabled)
                group_item.setData(Qt.UserRole, "")
                item_list.addItem(group_item)
                for field_name, label in group_items:
                    add_checkable_item(field_name, label)
        else:
            for field_name, label in items or []:
                add_checkable_item(field_name, label)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        button_box.rejected.connect(dialog.reject)

        def selected_items() -> list[str]:
            return [
                str(item_list.item(index).data(Qt.UserRole))
                for index in range(item_list.count())
                if item_list.item(index).data(Qt.UserRole)
                and item_list.item(index).checkState() == Qt.Checked
            ]

        def accept_if_any_selected() -> None:
            if not selected_items():
                self._show_warning(empty_title, empty_message)
                return
            dialog.accept()

        ok_button = button_box.button(QDialogButtonBox.Ok)
        if ok_button is not None:
            ok_button.clicked.connect(accept_if_any_selected)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(item_list)
        layout.addWidget(button_box)
        dialog.resize(360, 360)

        if dialog.exec_() != QDialog.Accepted:
            return None
        return selected_items()

    def _write_project_excel(self, output_path: Path, selected_fieldnames: list[str]) -> None:
        if self.project_document is None:
            raise ValueError("Нет проекта для экспорта.")

        rows = [self._project_export_row(record) for record in self.project_document.images]
        selected_headers = [PROJECT_EXPORT_COLUMN_LABELS[field_name] for field_name in selected_fieldnames]
        localized_rows = [
            {
                PROJECT_EXPORT_COLUMN_LABELS[field_name]: row.get(field_name, "")
                for field_name in selected_fieldnames
            }
            for row in rows
        ]
        write_xlsx_table(output_path, selected_headers, localized_rows, sheet_name="Сводка")

    def export_image_properties_excel(self) -> None:
        if self.project_document is None:
            self._show_warning("Нет проекта", "Сначала создайте или откройте проект.")
            return

        record = self._selected_project_image()
        if record is None:
            self._show_warning("Нет выбранного изображения", "Выберите изображение в списке проекта.")
            return

        self._save_current_project_annotation()
        self._update_project_properties()
        selected_properties = self._select_image_property_export_items()
        if selected_properties is None:
            return
        if not selected_properties:
            self._show_warning("Нет выбранных свойств", "Выберите хотя бы одно свойство для экспорта.")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт свойств изображения в Excel",
            str(self._default_image_properties_export_path(record)),
            "Excel (*.xlsx);;All files (*)",
        )
        if not file_name:
            return

        output_path = _ensure_xlsx_suffix(Path(file_name))
        try:
            self._write_image_properties_excel(output_path, selected_properties)
        except OSError as exc:
            self._show_error(
                "Ошибка экспорта свойств изображения в Excel",
                f"Не удалось сохранить Excel-файл: {exc}",
            )
            return

        self.statusBar().showMessage(f"Экспорт свойств изображения в Excel выполнен: {output_path.name}")

    def _select_image_property_export_items(self) -> list[str] | None:
        return self._select_checked_export_items(
            title="Свойства экспорта",
            grouped_items=IMAGE_PROPERTY_GROUPS,
            empty_title="Нет выбранных свойств",
            empty_message="Выберите хотя бы одно свойство для экспорта.",
        )

    def _default_image_properties_export_path(self, record: ProjectImageRecord) -> Path:
        stem = Path(record.display_name).stem or "image"
        file_name = f"{stem}_properties.xlsx"
        if self.project_path is None:
            return Path(file_name)
        return self.project_path.parent / file_name

    def _write_image_properties_excel(self, output_path: Path, selected_properties: list[str]) -> None:
        record = self._selected_project_image()
        if record is None:
            raise ValueError("Нет выбранного изображения для экспорта.")

        values = self._image_property_export_values(record)
        rows = [
            {
                "Свойство": IMAGE_PROPERTY_EXPORT_LABELS[property_key],
                "Значение": values.get(property_key, ""),
            }
            for property_key in selected_properties
        ]
        write_xlsx_table(output_path, ["Свойство", "Значение"], rows, sheet_name="Свойства")

    def _image_property_export_values(self, record: ProjectImageRecord) -> dict[str, str]:
        self._normalize_record_metadata(record)
        red = self.property_red.text()
        green = self.property_green.text()
        blue = self.property_blue.text()
        average_color = "-"
        if all(value not in ("", "-") for value in (red, green, blue)):
            average_color = f"RGB({red}, {green}, {blue})"
        return {
            "id": self.image_id.text().strip(),
            "file_name": self.property_file_name.text().strip(),
            "relative_path": self.property_path.text(),
            "size": self.property_size.text(),
            "annotation": self.property_annotation.text(),
            "point_count": self.property_points.text(),
            "contour_pixel_count": self.property_contour_pixels.text(),
            "red": red,
            "green": green,
            "blue": blue,
            "average_color": average_color,
            "status": self.property_status.text(),
            "added_at": self.metadata_added_at.text().strip(),
            "captured_at": self.metadata_captured_at.text().strip(),
            "illumination": self.metadata_illumination.text().strip(),
            "humidity": self.metadata_humidity.text().strip(),
            "wind_speed": self.metadata_wind_speed.text().strip(),
            "wind_direction": self.metadata_wind_direction.text().strip(),
            "latitude": self.metadata_latitude.text().strip(),
            "longitude": self.metadata_longitude.text().strip(),
            "diagnosis": self.metadata_diagnosis.text().strip(),
            "notes": self.metadata_notes.toPlainText(),
        }

    def _project_export_row(self, record: ProjectImageRecord) -> dict[str, str]:
        self._normalize_record_metadata(record)
        row = {
            "id": record.id,
            "file_name": record.display_name,
            "relative_path": record.relative_path,
            "has_annotation": "1" if record.annotation is not None else "0",
            "status": "ok",
            "diagnosis": str(record.metadata.get("diagnosis", "")),
            "r": "",
            "g": "",
            "b": "",
            "contour_pixel_count": "",
        }

        image_path = self._project_image_path(record)
        if not image_path.exists():
            row["status"] = "файл не найден"
            return row
        if record.annotation is None:
            row["status"] = "нет контура"
            return row

        try:
            loaded_image = load_raster_image(image_path)
            rgb_pixels = _annotation_rgb_pixels(loaded_image.rgb_array, record.annotation)
        except (OSError, ValueError) as exc:
            row["status"] = f"ошибка: {exc}"
            return row

        if rgb_pixels.size == 0:
            row["status"] = "внутри контура нет пикселов"
            return row

        mean_values = np.rint(rgb_pixels.mean(axis=0)).astype(int)
        row["r"] = str(int(mean_values[0]))
        row["g"] = str(int(mean_values[1]))
        row["b"] = str(int(mean_values[2]))
        row["contour_pixel_count"] = str(int(rgb_pixels.shape[0]))
        return row

    def create_new_contour(self) -> None:
        if self._selected_project_image() is None or not self.canvas.has_image():
            self._show_warning("Нет изображения", "Сначала выберите изображение проекта.")
            return

        node_count, accepted = QInputDialog.getInt(
            self,
            "Новый контур",
            "Количество узлов контура:",
            5,
            3,
            128,
            1,
        )
        if not accepted:
            return

        record = self._selected_project_image()
        has_saved_project_contour = (
            record is not None
            and (record.annotation is not None or record.raw_annotation is not None)
        )
        if self.canvas.has_contour() or has_saved_project_contour:
            answer = QMessageBox.question(
                self,
                "Заменить контур?",
                "Текущий контур будет заменён новым окружным контуром. Продолжить?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        image_size = self.canvas.image_size()
        if image_size is None:
            self._show_warning("Нет изображения", "Сначала выберите изображение проекта.")
            return

        width, height = image_size
        points = _circle_contour_points(width, height, node_count)
        try:
            self.canvas.set_contour(points)
        except ValueError as exc:
            self._show_error("Не удалось создать контур", str(exc))
            return

        self._save_current_project_annotation()
        self._refresh_histograms()
        self._update_project_properties()
        self._update_action_states()
        self.statusBar().showMessage(f"Создан новый контур: {len(points)} узлов.")

    def detect_contour(self) -> None:
        if self._selected_project_image() is None or not self.canvas.has_image():
            self._show_warning("Нет изображения", "Сначала выберите изображение проекта.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            points = detect_leaf_contour(self.canvas.current_rgb_array())
            self.canvas.set_contour(points)
        except ValueError as exc:
            self._show_error("Не удалось определить контур", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.statusBar().showMessage(f"Контур построен: {len(points)} узлов.")

    def delete_current_contour(self) -> None:
        record = self._selected_project_image()
        has_project_contour = (
            record is not None
            and (record.annotation is not None or record.raw_annotation is not None)
        )
        if not self.canvas.has_contour() and not has_project_contour:
            self._show_warning("Нет контура", "Для текущего изображения нет контура.")
            return

        if record is None:
            self._show_warning("Изображение не выбрано", "Выберите изображение в списке проекта.")
            return

        record.annotation = None
        record.raw_annotation = None
        record.annotation_error = None

        if self._current_project_image_id == record.id and self.canvas.has_contour():
            self.canvas.clear_contour()
        else:
            self._update_current_project_list_item(record)
            self._update_project_summary_properties()
            self._update_project_properties()
            self._update_action_states()
            self._schedule_project_save()

        self._clear_histograms("Создайте основной контур, чтобы увидеть гистограмму.")
        self._save_project_silently(show_error=True)
        self.statusBar().showMessage(f"Контур удалён: {record.display_name}")

    def flatten_background(self) -> None:
        if self._selected_project_image() is None or not self.canvas.has_image():
            self._show_warning("Нет изображения", "Сначала выберите изображение проекта.")
            return
        if not self.canvas.has_contour():
            self._show_warning("Нет контура", "Сначала постройте или загрузите контур.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.canvas.flatten_background_to_white()
        except ValueError as exc:
            self._show_error("Не удалось выровнять фон", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.statusBar().showMessage("Фон за пределами контура выровнен до белого.")

    def save_annotation_file(self) -> None:
        if not self.canvas.has_contour():
            self._show_warning("Нет контура", "Сначала постройте или загрузите контур.")
            return
        self._save_current_project_annotation()

        initial_name = self._default_annotation_file_name()
        if self._current_annotation_path is not None:
            initial_name = str(self._current_annotation_path)

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить аннотацию",
            initial_name,
            "JSON (*.json);;All files (*)",
        )
        if not file_name:
            return

        annotation_path = _ensure_json_suffix(Path(file_name))
        annotation = self._build_annotation(annotation_path)

        try:
            save_annotation(annotation_path, annotation)
        except OSError as exc:
            self._show_error("Ошибка сохранения", f"Не удалось сохранить аннотацию: {exc}")
            return

        self._current_annotation_path = annotation_path
        self.statusBar().showMessage(f"Аннотация сохранена: {annotation_path.name}")

    def open_annotation_file(self) -> None:
        if self.project_document is None or self._selected_project_image() is None:
            self._show_warning("Нет выбранного изображения", "Сначала выберите изображение проекта.")
            return

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть аннотацию",
            "",
            "JSON (*.json);;All files (*)",
        )
        if not file_name:
            return

        annotation_path = Path(file_name)
        try:
            annotation = load_annotation(annotation_path)
        except (OSError, ValueError) as exc:
            self._show_error("Ошибка загрузки", str(exc))
            return

        if not self._prepare_image_for_annotation(annotation, annotation_path):
            return

        if self.canvas.image_size() != (annotation.image_width, annotation.image_height):
            self._show_error(
                "Несовпадение размеров",
                "Размеры изображения не совпадают с данными аннотации.",
            )
            return

        try:
            self.canvas.set_contour(annotation.points)
        except ValueError as exc:
            self._show_error("Ошибка загрузки контура", str(exc))
            return

        self._current_annotation_path = annotation_path
        self.statusBar().showMessage(f"Аннотация открыта: {annotation_path.name}")

    def _set_project(self, project_path: Path, document: ProjectDocument) -> None:
        self.project_path = project_path.resolve()
        self.project_document = document
        self._current_project_image_id = None
        self._current_annotation_path = None
        self._refresh_project_list()
        if document.images:
            self._select_project_image(document.images[0].id)
        else:
            self._clear_current_image_display()
        self._update_project_panel()
        self._update_window_title()

    def _clear_project_state(self) -> None:
        self._project_autosave_timer.stop()
        self.project_document = None
        self.project_path = None
        self._current_project_image_id = None
        self._current_annotation_path = None
        self._refresh_project_list()
        self._clear_current_image_display()
        self._update_project_panel()
        self._update_window_title()

    def _clear_current_image_display(self) -> None:
        self._loading_project_image = True
        try:
            self.canvas.clear_image()
        finally:
            self._loading_project_image = False
        self._clear_histograms("Откройте изображение и создайте контур, чтобы увидеть гистограмму.")
        self._update_project_properties()
        self._update_action_states()

    def _update_project_panel(self) -> None:
        has_project = self.project_document is not None
        self.project_list.setEnabled(has_project)
        self.export_image_properties_excel_button.setEnabled(has_project and self._selected_project_image() is not None)
        self._update_project_summary_properties()
        self._update_project_properties()
        self._update_action_states()

    def _refresh_project_list(self) -> None:
        self._updating_project_list = True
        try:
            selected_id = self._current_project_image_id
            self.project_list.clear()
            if self.project_document is None:
                return
            for record in self.project_document.images:
                item = QListWidgetItem()
                item.setData(Qt.UserRole, record.id)
                self._populate_project_list_item(item, record)
                self.project_list.addItem(item)
            if selected_id is not None:
                self._select_project_image(selected_id, emit_signal=False)
        finally:
            self._updating_project_list = False

    def _populate_project_list_item(self, item: QListWidgetItem, record: ProjectImageRecord) -> None:
        details: list[str] = []
        if record.annotation is not None:
            details.append("аннотация есть")
        elif record.annotation_error:
            details.append("ошибка аннотации")
        else:
            details.append("без аннотации")

        image_path = self._project_image_path(record)
        file_is_missing = self.project_path is not None and not image_path.exists()
        has_valid_annotation = record.annotation is not None
        if file_is_missing:
            details.append("файл не найден")

        if file_is_missing or not has_valid_annotation:
            item.setForeground(QColor("#b42318"))
        else:
            item.setForeground(QColor("#1f2937"))

        item.setText(f"{record.display_name}\n{'; '.join(details)}")
        item.setToolTip(f"{record.display_name}\n{record.relative_path}\n{'; '.join(details)}")

    def _update_current_project_list_item(self, record: ProjectImageRecord) -> None:
        current_item = self.project_list.currentItem()
        if current_item is None or current_item.data(Qt.UserRole) != record.id:
            return
        self._populate_project_list_item(current_item, record)

    def _select_project_image(self, record_id: str, *, emit_signal: bool = True) -> None:
        previous_state = self._updating_project_list
        if not emit_signal:
            self._updating_project_list = True
        try:
            for row in range(self.project_list.count()):
                item = self.project_list.item(row)
                if item.data(Qt.UserRole) == record_id:
                    self.project_list.setCurrentRow(row)
                    if emit_signal:
                        return
                    self._current_project_image_id = record_id
                    return
            self.project_list.clearSelection()
            self._current_project_image_id = None
        finally:
            self._updating_project_list = previous_state

    def _handle_project_selection_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if self._updating_project_list:
            return

        selected_id = current.data(Qt.UserRole) if current is not None else None
        if selected_id == self._current_project_image_id:
            return

        self._save_current_project_annotation()
        self._current_project_image_id = str(selected_id) if selected_id else None
        self._current_annotation_path = None

        record = self._selected_project_image()
        if record is None:
            self._clear_current_image_display()
            return

        self._load_project_image(record)

    def _load_project_image(self, record: ProjectImageRecord) -> None:
        image_path = self._project_image_path(record)
        self._loading_project_image = True
        size_changed = False
        loaded_ok = False
        try:
            if not image_path.exists():
                self.canvas.clear_image()
                self._clear_histograms("Файл изображения из проекта не найден.")
                self.statusBar().showMessage(f"Файл изображения не найден: {record.display_name}")
            else:
                try:
                    loaded_image = load_raster_image(image_path)
                except (OSError, ValueError) as exc:
                    self.canvas.clear_image()
                    self._clear_histograms(str(exc))
                    self.statusBar().showMessage(f"Не удалось открыть изображение: {record.display_name}")
                else:
                    loaded_ok = True
                    self.canvas.set_loaded_image(loaded_image)
                    if (record.image_width, record.image_height) != (
                        loaded_image.width,
                        loaded_image.height,
                    ):
                        record.image_width = loaded_image.width
                        record.image_height = loaded_image.height
                        size_changed = True

                    if record.annotation is not None:
                        if (record.annotation.image_width, record.annotation.image_height) != (
                            loaded_image.width,
                            loaded_image.height,
                        ):
                            self.statusBar().showMessage(
                                "Аннотация не загружена: размеры изображения не совпадают."
                            )
                        else:
                            try:
                                self.canvas.set_contour(record.annotation.points)
                            except ValueError as exc:
                                record.annotation_error = str(exc)
                                self.statusBar().showMessage("Аннотация повреждена и не загружена.")
                    elif record.annotation_error:
                        self.statusBar().showMessage("У выбранного изображения повреждена аннотация.")
        finally:
            self._loading_project_image = False

        if size_changed:
            self._schedule_project_save()
        if loaded_ok:
            self._refresh_histograms()
        self._update_current_project_list_item(record)
        self._update_project_summary_properties()
        self._update_project_properties()
        self._update_action_states()
        self._update_window_title()

    def _handle_contour_geometry_changed(self) -> None:
        if self._loading_project_image:
            return
        self._save_current_project_annotation()
        self._update_project_summary_properties()
        self._update_project_properties()

    def _save_current_project_annotation(self) -> None:
        if self._loading_project_image:
            return
        if self.project_document is None:
            return

        record = self._current_project_image()
        if record is None or not self.canvas.has_image():
            return

        image_size = self.canvas.image_size()
        if image_size is None:
            return

        width, height = image_size
        record.image_width = width
        record.image_height = height
        if self.canvas.has_contour():
            record.annotation = Annotation(
                image_path=record.relative_path,
                image_width=width,
                image_height=height,
                points=self.canvas.contour_points(),
                closed=True,
            )
            record.annotation_error = None
            record.raw_annotation = None
        else:
            record.annotation = None
            record.raw_annotation = None

        self._update_current_project_list_item(record)
        self._schedule_project_save()

    def _schedule_project_save(self) -> None:
        if self.project_document is None or self.project_path is None:
            return
        self._project_autosave_timer.start(350)

    def _save_project_silently(self, *, show_error: bool = False) -> bool:
        if self.project_document is None or self.project_path is None:
            return True
        try:
            save_project(self.project_path, self.project_document)
        except OSError as exc:
            if show_error:
                self._show_error("Ошибка сохранения проекта", f"Не удалось сохранить проект: {exc}")
            return False
        return True

    def _selected_project_image_id(self) -> str | None:
        item = self.project_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.UserRole)
        return str(value) if value else None

    def _selected_project_image(self) -> ProjectImageRecord | None:
        selected_id = self._selected_project_image_id()
        if selected_id is None:
            return None
        return self._project_image_by_id(selected_id)

    def _current_project_image(self) -> ProjectImageRecord | None:
        if self._current_project_image_id is None:
            return None
        return self._project_image_by_id(self._current_project_image_id)

    def _project_image_by_id(self, record_id: str) -> ProjectImageRecord | None:
        if self.project_document is None:
            return None
        for record in self.project_document.images:
            if record.id == record_id:
                return record
        return None

    def _project_image_path(self, record: ProjectImageRecord) -> Path:
        if self.project_path is None:
            return Path(record.relative_path)
        return (self.project_path.parent / record.relative_path).resolve()

    def _update_project_summary_properties(self) -> None:
        has_project = self.project_document is not None
        self.project_properties_empty_label.setVisible(not has_project)
        self.project_properties_browser.setVisible(has_project)
        self.project_general_info.setEnabled(has_project)

        if self.project_document is None:
            self.project_properties_empty_label.setText("Создайте или откройте проект.")
            self.project_image_count.setText("-")
            self.project_mean_red.setText("-")
            self.project_mean_green.setText("-")
            self.project_mean_blue.setText("-")
            self._updating_project_info_fields = True
            try:
                with QSignalBlocker(self.project_general_info):
                    self.project_general_info.setPlainText("")
            finally:
                self._updating_project_info_fields = False
            return

        self._updating_project_info_fields = True
        try:
            with QSignalBlocker(self.project_general_info):
                self.project_general_info.setPlainText(self.project_document.project_info.general_info)
        finally:
            self._updating_project_info_fields = False

        self.project_image_count.setText(str(len(self.project_document.images)))
        mean_rgb = self._project_contours_mean_rgb()
        if mean_rgb is None:
            red_text = green_text = blue_text = "-"
        else:
            red, green, blue = mean_rgb
            red_text = str(red)
            green_text = str(green)
            blue_text = str(blue)
        self.project_mean_red.setText(red_text)
        self.project_mean_green.setText(green_text)
        self.project_mean_blue.setText(blue_text)

    def _project_contours_mean_rgb(self) -> tuple[int, int, int] | None:
        if self.project_document is None:
            return None

        total = np.zeros(3, dtype=np.float64)
        pixel_count = 0
        for record in self.project_document.images:
            if record.annotation is None or record.annotation_error:
                continue

            image_path = self._project_image_path(record)
            if not image_path.exists():
                continue

            try:
                loaded_image = load_raster_image(image_path)
            except (OSError, ValueError):
                continue

            if (record.annotation.image_width, record.annotation.image_height) != (
                loaded_image.width,
                loaded_image.height,
            ):
                continue

            pixels = _annotation_rgb_pixels(loaded_image.rgb_array, record.annotation)
            if pixels.size == 0:
                continue

            total += pixels.sum(axis=0)
            pixel_count += int(pixels.shape[0])

        if pixel_count == 0:
            return None

        mean_values = np.rint(total / pixel_count).astype(int)
        return int(mean_values[0]), int(mean_values[1]), int(mean_values[2])

    def _update_project_properties(self, *_args) -> None:
        record = self._selected_project_image()
        has_record = record is not None
        self.properties_empty_label.setVisible(not has_record)
        self.properties_browser.setVisible(has_record)
        self.export_image_properties_excel_button.setEnabled(has_record)
        self.rename_file_as_id_button.setEnabled(has_record)
        self.generate_image_id_button.setEnabled(has_record)

        if record is None:
            self.properties_empty_label.setText(
                "Создайте или откройте проект, затем выберите изображение."
                if self.project_document is None
                else "Изображение не выбрано."
            )
            self._set_average_color_swatch(None)
            return

        self._normalize_record_metadata(record)
        image_path = self._project_image_path(record)
        file_exists = image_path.exists()
        size_text = "-"
        if record.image_width and record.image_height:
            size_text = f"{record.image_width} x {record.image_height}"

        annotation_text = "нет"
        point_count_text = "-"
        if record.annotation is not None:
            annotation_text = "есть"
            point_count_text = str(record.point_count())
        elif record.annotation_error:
            annotation_text = f"ошибка: {record.annotation_error}"

        contour_stats = self._current_contour_stats(record)
        if contour_stats is None:
            red_text = green_text = blue_text = "-"
            contour_pixels_text = "-"
            mean_rgb = None
        else:
            mean_rgb, contour_pixel_count = contour_stats
            red, green, blue = mean_rgb
            red_text = str(red)
            green_text = str(green)
            blue_text = str(blue)
            contour_pixels_text = str(contour_pixel_count)

        self.property_file_name.setText(record.display_name)
        self.property_path.setText(record.relative_path)
        self.property_size.setText(size_text)
        self.property_annotation.setText(annotation_text)
        self.property_points.setText(point_count_text)
        self.property_contour_pixels.setText(contour_pixels_text)
        self.property_red.setText(red_text)
        self.property_green.setText(green_text)
        self.property_blue.setText(blue_text)
        self.property_status.setText("найден" if file_exists else "отсутствует")
        self._set_average_color_swatch(mean_rgb)
        self._load_metadata_fields(record)

    def _normalize_record_metadata(self, record: ProjectImageRecord) -> None:
        normalized_metadata = default_project_image_metadata()
        normalized_metadata.update(record.metadata)
        record.metadata = normalized_metadata

    def _load_metadata_fields(self, record: ProjectImageRecord) -> None:
        self._updating_metadata_fields = True
        try:
            with QSignalBlocker(self.property_file_name):
                self.property_file_name.setText(record.display_name)
            with QSignalBlocker(self.image_id):
                self.image_id.setText(record.id)
            for metadata_key, field in self._metadata_fields.items():
                with QSignalBlocker(field):
                    field.setText(str(record.metadata.get(metadata_key, "")))
            with QSignalBlocker(self.metadata_notes):
                self.metadata_notes.setPlainText(str(record.metadata.get("notes", "")))
        finally:
            self._updating_metadata_fields = False

    def _handle_project_general_info_changed(self) -> None:
        if self._updating_project_info_fields:
            return
        if self.project_document is None:
            return

        value = self.project_general_info.toPlainText()
        if self.project_document.project_info.general_info == value:
            return

        self.project_document.project_info.general_info = value
        self._schedule_project_save()

    def _handle_metadata_line_edit_finished(self, metadata_key: str, field: QLineEdit) -> None:
        if self._updating_metadata_fields:
            return

        self._store_metadata_text_value(metadata_key, field.text().strip(), field)

    def _handle_image_id_edit_finished(self) -> None:
        if self._updating_metadata_fields:
            return

        self._store_record_id_value(self.image_id.text().strip(), self.image_id)

    def _store_metadata_text_value(
        self,
        metadata_key: str,
        value: str,
        field: QLineEdit | None = None,
    ) -> bool:
        record = self._selected_project_image()
        if record is None:
            return False

        validation_error = _metadata_validation_error(metadata_key, value)
        if validation_error:
            self._show_warning("Некорректное значение", validation_error)
            if field is not None:
                with QSignalBlocker(field):
                    field.setText(str(record.metadata.get(metadata_key, "")))
            return False

        self._normalize_record_metadata(record)
        if str(record.metadata.get(metadata_key, "")) == value:
            return True

        record.set_metadata_value(metadata_key, value)
        if field is not None:
            with QSignalBlocker(field):
                field.setText(value)
        self._schedule_project_save()
        return True

    def _store_record_id_value(self, value: str, field: QLineEdit | None = None) -> bool:
        record = self._selected_project_image()
        if record is None:
            return False

        normalized_value = value.strip()
        validation_error = self._record_id_validation_error(record, normalized_value)
        if validation_error:
            self._show_warning("Некорректный ID", validation_error)
            if field is not None:
                with QSignalBlocker(field):
                    field.setText(record.id)
            return False

        if record.id == normalized_value:
            if field is not None:
                with QSignalBlocker(field):
                    field.setText(normalized_value)
            return True

        old_id = record.id
        record.id = normalized_value
        if self._current_project_image_id == old_id:
            self._current_project_image_id = normalized_value

        current_item = self.project_list.currentItem()
        if current_item is not None and current_item.data(Qt.UserRole) == old_id:
            current_item.setData(Qt.UserRole, normalized_value)
            self._populate_project_list_item(current_item, record)

        if field is not None:
            with QSignalBlocker(field):
                field.setText(normalized_value)
        self._schedule_project_save()
        return True

    def _record_id_validation_error(self, current_record: ProjectImageRecord, value: str) -> str:
        if not value:
            return "ID изображения не должен быть пустым."
        if self.project_document is None:
            return ""

        normalized_value = value.strip()
        for record in self.project_document.images:
            if record.id == current_record.id:
                continue
            if record.id == normalized_value:
                return "Такой ID уже используется другим изображением проекта."
        return ""

    def _new_project_image_id(self) -> str:
        used_ids = set()
        if self.project_document is not None:
            used_ids = {record.id for record in self.project_document.images}

        for _attempt in range(10):
            candidate = str(uuid4())
            if candidate not in used_ids:
                return candidate
        raise ValueError("Не удалось сгенерировать уникальный ID изображения.")

    def _generate_image_id(self) -> None:
        record = self._selected_project_image()
        if record is None:
            return

        try:
            generated_id = self._new_project_image_id()
        except ValueError as exc:
            self._show_error("Ошибка генерации ID", str(exc))
            return

        if not self._store_record_id_value(generated_id, self.image_id):
            return

        self.statusBar().showMessage(f"Сгенерирован новый ID изображения: {generated_id}")

    def _open_metadata_datetime_dialog(self, metadata_key: str, field: QLineEdit) -> None:
        if self._updating_metadata_fields:
            return
        if self._selected_project_image() is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Выбор даты и времени")

        editor = QDateTimeEdit(_qdatetime_from_text(field.text()), dialog)
        editor.setCalendarPopup(True)
        editor.setDisplayFormat("yyyy-MM-dd HH:mm:ss")

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        clear_button = button_box.addButton("Очистить", QDialogButtonBox.ActionRole)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(editor)
        layout.addWidget(button_box)

        clear_requested = False

        def clear_value() -> None:
            nonlocal clear_requested
            clear_requested = True
            dialog.accept()

        clear_button.clicked.connect(clear_value)

        if dialog.exec_() != QDialog.Accepted:
            return

        value = ""
        if not clear_requested:
            value = editor.dateTime().toPyDateTime().isoformat(timespec="seconds")
        self._store_metadata_text_value(metadata_key, value, field)

    def _handle_metadata_notes_changed(self) -> None:
        if self._updating_metadata_fields:
            return

        record = self._selected_project_image()
        if record is None:
            return

        self._normalize_record_metadata(record)
        notes = self.metadata_notes.toPlainText()
        if str(record.metadata.get("notes", "")) == notes:
            return

        record.set_metadata_value("notes", notes)
        self._schedule_project_save()

    def _handle_file_name_edit_finished(self) -> None:
        if self._updating_metadata_fields:
            return

        record = self._selected_project_image()
        if record is None:
            return

        new_name = self.property_file_name.text().strip()
        if new_name == record.display_name:
            return

        if self._rename_project_image(record, new_name):
            self._update_current_project_list_item(record)
            self._update_project_properties()
            self._update_window_title()
            self._save_project_silently(show_error=True)
            return

        with QSignalBlocker(self.property_file_name):
            self.property_file_name.setText(record.display_name)

    def _rename_file_to_image_id(self) -> None:
        record = self._selected_project_image()
        if record is None:
            return

        image_id = self.image_id.text().strip()
        if not self._store_record_id_value(image_id, self.image_id):
            return

        image_id = record.id.strip()
        if not image_id:
            self._show_warning("Некорректный ID", "ID изображения не должен быть пустым.")
            return
        if "/" in image_id or "\\" in image_id:
            self._show_warning(
                "Некорректный ID",
                "ID изображения не должен содержать символы пути.",
            )
            return

        old_suffix = Path(record.relative_path).suffix
        target_name = image_id
        if old_suffix and not target_name.lower().endswith(old_suffix.lower()):
            target_name = f"{target_name}{old_suffix}"

        if target_name == record.display_name:
            self.statusBar().showMessage("Файл уже назван как ID.")
            return

        if self._rename_project_image(record, target_name):
            self._update_current_project_list_item(record)
            self._update_project_properties()
            self._update_window_title()
            self._save_project_silently(show_error=True)

    def _rename_project_image(self, record: ProjectImageRecord, new_name: str) -> bool:
        if not new_name:
            self._show_warning("Некорректное имя", "Имя файла не должно быть пустым.")
            return False

        new_file_name = Path(new_name).name
        if not new_file_name:
            self._show_warning("Некорректное имя", "Имя файла не должно быть пустым.")
            return False

        old_relative_path = Path(record.relative_path)
        old_suffix = old_relative_path.suffix
        candidate = Path(new_file_name)
        if not candidate.suffix:
            candidate = candidate.with_suffix(old_suffix)

        if candidate.suffix.lower() not in SUPPORTED_RASTER_SUFFIXES:
            self._show_warning(
                "Некорректное расширение",
                "Имя файла должно иметь поддерживаемое расширение изображения.",
            )
            return False

        image_path = self._project_image_path(record)
        if self.project_path is None:
            record.display_name = candidate.name
            return True

        if not image_path.exists():
            record.display_name = candidate.name
            self._show_warning(
                "Файл отсутствует",
                "Файл изображения не найден на диске. Изменено только отображаемое имя.",
            )
            self._schedule_project_save()
            return True

        target_path = image_path.with_name(candidate.name)
        if target_path.resolve() != image_path.resolve() and target_path.exists():
            self._show_warning(
                "Имя занято",
                "Файл с таким именем уже существует в папке проекта.",
            )
            return False

        try:
            if target_path.resolve() != image_path.resolve():
                image_path.rename(target_path)
        except OSError as exc:
            self._show_error("Ошибка переименования", f"Не удалось переименовать файл: {exc}")
            return False

        record.relative_path = portable_path_reference(target_path, self.project_path.parent)
        record.display_name = candidate.name
        if record.annotation is not None:
            record.annotation.image_path = record.relative_path
        self._schedule_project_save()
        self.statusBar().showMessage(f"Файл переименован: {record.display_name}")
        return True

    def _current_contour_stats(
        self,
        record: ProjectImageRecord,
    ) -> tuple[tuple[int, int, int], int] | None:
        if record.id != self._current_project_image_id:
            return None
        if not self.canvas.has_image() or not self.canvas.has_contour():
            return None
        try:
            pixels = self.canvas.contour_rgb_pixels()
        except ValueError:
            return None
        if pixels.size == 0:
            return None
        mean_values = np.rint(pixels.mean(axis=0)).astype(int)
        mean_rgb = int(mean_values[0]), int(mean_values[1]), int(mean_values[2])
        return mean_rgb, int(pixels.shape[0])

    def _set_average_color_swatch(self, rgb: tuple[int, int, int] | None) -> None:
        if rgb is None:
            self.average_color_swatch.setStyleSheet(
                "QFrame#averageColorSwatch { border: 1px solid #95a3b8; border-radius: 4px; background: transparent; }"
            )
            return
        red, green, blue = rgb
        self.average_color_swatch.setStyleSheet(
            "QFrame#averageColorSwatch { "
            f"border: 1px solid #95a3b8; border-radius: 4px; background: rgb({red}, {green}, {blue}); "
            "}"
        )

    def show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "О программе",
            (
                "MagicBorder\n\n"
                "Локальное приложение для выделения и редактирования контура листа растения.\n\n"
                "Разработчик: Дьяченко Р.А.\n\n"
                "Подсказки:\n"
                "- перетаскивайте узлы мышью;\n"
                "- двойной щелчок рядом с сегментом добавляет узел;\n"
                "- правая кнопка мыши по узлу или клавиша Delete удаляет узел;\n"
                "- Ctrl + колесо мыши изменяет масштаб."
            ),
        )

    def _build_annotation(self, annotation_path: Path) -> Annotation:
        image_path = self.canvas.current_image_path()
        image_size = self.canvas.image_size()
        if image_path is None or image_size is None:
            raise ValueError("Нет изображения для создания аннотации.")

        width, height = image_size
        return Annotation(
            image_path=portable_path_reference(image_path, annotation_path.parent),
            image_width=width,
            image_height=height,
            points=self.canvas.contour_points(),
            closed=True,
        )

    def _prepare_image_for_annotation(self, annotation: Annotation, annotation_path: Path) -> bool:
        current_image_size = self.canvas.image_size()
        if current_image_size == (annotation.image_width, annotation.image_height):
            return True

        self._show_warning(
            "Аннотация не подходит",
            "Аннотацию можно загрузить только для выбранного изображения проекта с тем же размером.",
        )
        return False

    def _update_window_title(self) -> None:
        if self.project_document is not None:
            record = self._current_project_image()
            if record is not None:
                self.setWindowTitle(f"{APP_TITLE} - {self.project_document.name} - {record.display_name}")
                return
            self.setWindowTitle(f"{APP_TITLE} - {self.project_document.name}")
            return

        self.setWindowTitle(APP_TITLE)

    def _default_annotation_file_name(self) -> str:
        image_path = self.canvas.current_image_path()
        if image_path is None:
            return "annotation.json"
        return str(image_path.with_suffix(".json"))

    def _default_histogram_file_name(self, color_space: str) -> str:
        fallback_name = f"{color_space}_histogram.png"
        image_path = self.canvas.current_image_path()
        if image_path is None:
            return fallback_name
        return str(image_path.with_name(f"{image_path.stem}_{color_space}_histogram.png"))

    def closeEvent(self, event) -> None:
        self._save_current_project_annotation()
        if self._save_project_silently(show_error=True):
            event.accept()
            return

        answer = QMessageBox.question(
            self,
            "Закрыть без сохранения?",
            "Проект не удалось сохранить. Закрыть приложение без сохранения проекта?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)


def _ensure_json_suffix(path: Path) -> Path:
    if path.suffix.lower() == ".json":
        return path
    if path.suffix:
        return path.with_suffix(".json")
    return path.with_suffix(".json")


def _ensure_xlsx_suffix(path: Path) -> Path:
    if path.suffix.lower() == ".xlsx":
        return path
    if path.suffix:
        return path.with_suffix(".xlsx")
    return path.with_suffix(".xlsx")


def _annotation_rgb_pixels(rgb_array: np.ndarray, annotation: Annotation) -> np.ndarray:
    mask = np.zeros(rgb_array.shape[:2], dtype=np.uint8)
    polygon = np.array(
        [
            [int(round(point.x)), int(round(point.y))]
            for point in annotation.points
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [polygon], 255)
    pixels = rgb_array[mask > 0]
    return np.ascontiguousarray(pixels.reshape((-1, 3)))


def _circle_contour_points(width: int, height: int, node_count: int) -> list[Point]:
    bounded_count = max(3, min(128, int(node_count)))
    image_width = max(1.0, float(width))
    image_height = max(1.0, float(height))
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    radius = max(0.5, 0.35 * min(image_width, image_height))
    radius = min(radius, max(0.5, min(center_x, center_y) - 0.5))

    points: list[Point] = []
    for index in range(bounded_count):
        angle = -math.pi / 2.0 + 2.0 * math.pi * index / bounded_count
        x = center_x + math.cos(angle) * radius
        y = center_y + math.sin(angle) * radius
        x = min(max(x, 0.0), image_width)
        y = min(max(y, 0.0), image_height)
        points.append(Point(x, y))
    return points


def _current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _metadata_validation_error(metadata_key: str, value: str) -> str:
    if not value:
        return ""

    if metadata_key in ("added_at", "captured_at"):
        if QDateTime.fromString(value, Qt.ISODate).isValid() or _datetime_from_text(value) is not None:
            return ""
        return "Дата и время должны быть указаны в формате ISO 8601 или ГГГГ-ММ-ДД ЧЧ:ММ:СС."

    numeric_ranges = {
        "humidity": (0.0, 100.0, "Влажность должна быть числом в диапазоне 0..100."),
        "wind_speed": (0.0, None, "Скорость ветра не должна быть отрицательной."),
        "latitude": (-90.0, 90.0, "Широта должна быть числом в диапазоне -90..90."),
        "longitude": (-180.0, 180.0, "Долгота должна быть числом в диапазоне -180..180."),
    }
    if metadata_key not in numeric_ranges:
        return ""

    minimum, maximum, message = numeric_ranges[metadata_key]
    try:
        number = float(value.replace(",", "."))
    except ValueError:
        return message

    if number < minimum:
        return message
    if maximum is not None and number > maximum:
        return message
    return ""


def _qdatetime_from_text(value: str) -> QDateTime:
    text = value.strip()
    if text:
        parsed = QDateTime.fromString(text, Qt.ISODate)
        if parsed.isValid():
            return parsed
        with_timezone = _datetime_from_text(text)
        if with_timezone is not None:
            return QDateTime(with_timezone)
    return QDateTime.currentDateTime()


def _datetime_from_text(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    for candidate in (text, text.replace(" ", "T", 1)):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    for format_string in ("%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S"):
        try:
            return datetime.strptime(text, format_string)
        except ValueError:
            continue
    return None


def _safe_project_name(raw_name: str) -> str:
    safe_name = "".join(
        char if char.isalnum() or char in ("-", "_", " ") else "_"
        for char in raw_name.strip()
    )
    return "_".join(safe_name.split())


def _project_relative_path_key(value: str) -> str:
    return str(value or "").replace("\\", "/").strip("/")


def _unique_destination_path(directory: Path, file_name: str) -> Path:
    source_name = Path(file_name).name
    stem = Path(source_name).stem or "image"
    suffix = Path(source_name).suffix
    candidate = directory / source_name
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{index}{suffix}"
        index += 1
    return candidate
