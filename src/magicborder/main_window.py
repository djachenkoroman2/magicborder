from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
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
from .icons import ACTION_VISUALS, TOOLBAR_ICON_SIZE, apply_action_visual
from .io_utils import image_open_filter, image_save_filter, load_annotation, load_raster_image, save_annotation
from .models import Annotation
from .path_utils import annotation_image_candidates, portable_path_reference

APP_TITLE = "MagicBorder"
WORKSPACE_DEFAULT_SIZES = [1120, 280]
HISTOGRAM_DEFAULT_SIZES = [170, 170, 170, 170, 170]


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
        self.canvas.contour_state_changed.connect(self._update_action_states)
        self.canvas.contour_geometry_changed.connect(self._schedule_histogram_refresh)
        self._update_action_states()
        self._refresh_histograms()

    def _create_workspace(self) -> QSplitter:
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
            "QSplitter::handle { background: #36404c; height: 7px; }"
            "QSplitter::handle:hover { background: #4b5968; }"
        )

        analysis_layout = QVBoxLayout(analysis_panel)
        analysis_layout.setContentsMargins(10, 10, 10, 10)
        analysis_layout.setSpacing(8)
        analysis_layout.addWidget(analysis_title)
        analysis_layout.addWidget(self.histogram_splitter, 1)

        analysis_panel.setStyleSheet(
            "QWidget#analysisPanel { background: #232933; border-left: 1px solid #343d49; }"
            "QLabel#analysisTitle { color: #eef3fa; font-size: 13px; font-weight: 600; }"
            "QFrame#histogramPanel { background: #202630; border: 1px solid #3a4451; border-radius: 6px; }"
            "QLabel#histogramTitle { color: #dfe7f0; font-size: 12px; font-weight: 600; }"
            "QToolButton { border: 1px solid transparent; border-radius: 4px; padding: 4px; }"
            "QToolButton:hover { background: #303947; border-color: #475568; }"
            "QToolButton:pressed { background: #3b4655; }"
        )

        self.workspace_splitter = QSplitter(Qt.Horizontal, self)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.addWidget(self.canvas)
        self.workspace_splitter.addWidget(analysis_panel)
        self.workspace_splitter.setStretchFactor(0, 4)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes(WORKSPACE_DEFAULT_SIZES)
        self.workspace_splitter.setStyleSheet(
            "QSplitter::handle { background: #303844; width: 8px; }"
            "QSplitter::handle:hover { background: #485665; }"
        )
        return self.workspace_splitter

    def _create_actions(self) -> None:
        self.open_image_action = QAction("Открыть изображение...", self)
        self.open_image_action.setShortcut("Ctrl+O")
        self.open_image_action.triggered.connect(self.open_image)

        self.save_image_action = QAction("Сохранить изображение...", self)
        self.save_image_action.setShortcut("Ctrl+S")
        self.save_image_action.triggered.connect(self.save_image)

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

        self.detect_contour_action = QAction("Определить контур", self)
        self.detect_contour_action.setShortcut("F5")
        self.detect_contour_action.triggered.connect(self.detect_contour)

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
            "open_image": self.open_image_action,
            "save_image": self.save_image_action,
            "exit": self.exit_action,
            "zoom_in": self.zoom_in_action,
            "zoom_out": self.zoom_out_action,
            "fit_image": self.fit_image_action,
            "actual_size": self.actual_size_action,
            "default_view": self.default_view_action,
            "detect_contour": self.detect_contour_action,
            "flatten_background": self.flatten_background_action,
            "save_annotation": self.save_annotation_action,
            "open_annotation": self.open_annotation_action,
            "about": self.about_action,
        }

        for action_name, action in action_map.items():
            apply_action_visual(action, ACTION_VISUALS[action_name])

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("Файл")
        file_menu.addAction(self.open_image_action)
        file_menu.addAction(self.save_image_action)
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
        tools_menu.addAction(self.detect_contour_action)
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

        toolbar.addAction(self.open_image_action)
        toolbar.addAction(self.save_image_action)
        toolbar.addAction(self.exit_action)
        toolbar.addSeparator()
        toolbar.addAction(self.zoom_in_action)
        toolbar.addAction(self.zoom_out_action)
        toolbar.addAction(self.fit_image_action)
        toolbar.addAction(self.actual_size_action)
        toolbar.addAction(self.default_view_action)
        toolbar.addSeparator()
        toolbar.addAction(self.detect_contour_action)
        toolbar.addAction(self.flatten_background_action)
        toolbar.addAction(self.save_annotation_action)
        toolbar.addAction(self.open_annotation_action)
        toolbar.addSeparator()
        toolbar.addAction(self.about_action)

    def _update_action_states(self, *_args) -> None:
        has_image = self.canvas.has_image()
        has_contour = self.canvas.has_contour()

        self.save_image_action.setEnabled(has_image)
        self.zoom_in_action.setEnabled(has_image)
        self.zoom_out_action.setEnabled(has_image)
        self.fit_image_action.setEnabled(has_image)
        self.actual_size_action.setEnabled(has_image)
        self.detect_contour_action.setEnabled(has_image)
        self.flatten_background_action.setEnabled(has_image and has_contour)
        self.save_annotation_action.setEnabled(has_image and has_contour)

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
        self.histogram_splitter.setSizes(HISTOGRAM_DEFAULT_SIZES)

    def open_image(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть изображение",
            "",
            image_open_filter(),
        )
        if not file_name:
            return
        self._load_image(Path(file_name))

    def save_image(self) -> None:
        if not self.canvas.has_image():
            self._show_warning("Нет изображения", "Сначала откройте изображение.")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить изображение",
            "",
            image_save_filter(),
        )
        if not file_name:
            return

        output_path = _ensure_image_suffix(Path(file_name))
        try:
            self.canvas.save_rendered_image(output_path)
        except ValueError as exc:
            self._show_error("Ошибка сохранения", str(exc))
            return

        self.statusBar().showMessage(f"Изображение сохранено: {output_path.name}")

    def detect_contour(self) -> None:
        if not self.canvas.has_image():
            self._show_warning("Нет изображения", "Сначала откройте изображение.")
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

    def flatten_background(self) -> None:
        if not self.canvas.has_image():
            self._show_warning("Нет изображения", "Сначала откройте изображение.")
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

    def _load_image(self, image_path: Path, *, reset_annotation: bool = True) -> bool:
        try:
            loaded_image = load_raster_image(image_path)
        except (OSError, ValueError) as exc:
            self._show_error("Ошибка открытия", str(exc))
            return False

        self.canvas.set_loaded_image(loaded_image)
        self._update_window_title()
        if reset_annotation:
            self._current_annotation_path = None
        return True

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

        resolved_path = self._resolve_annotation_image_path(annotation, annotation_path)
        if resolved_path is None:
            self._show_warning(
                "Не найдено изображение",
                "Исходное изображение из аннотации не найдено. Откройте соответствующий файл вручную и повторите загрузку аннотации.",
            )
            return False

        if not self._load_image(resolved_path, reset_annotation=False):
            return False
        return True

    def _resolve_annotation_image_path(self, annotation: Annotation, annotation_path: Path) -> Path | None:
        if not annotation.image_path:
            return None

        for candidate in annotation_image_candidates(annotation.image_path, annotation_path):
            if candidate.exists():
                return candidate
        return None

    def _update_window_title(self) -> None:
        image_path = self.canvas.current_image_path()
        if image_path is None:
            self.setWindowTitle(APP_TITLE)
            return
        self.setWindowTitle(f"{APP_TITLE} - {image_path.name}")

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

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)


def _ensure_image_suffix(path: Path) -> Path:
    if path.suffix:
        return path
    return path.with_suffix(".png")


def _ensure_json_suffix(path: Path) -> Path:
    if path.suffix.lower() == ".json":
        return path
    if path.suffix:
        return path.with_suffix(".json")
    return path.with_suffix(".json")
