from __future__ import annotations

import os
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QToolBar,
)

from .canvas import ImageCanvas
from .detector import detect_leaf_contour
from .icons import ACTION_VISUALS, TOOLBAR_ICON_SIZE, apply_action_visual
from .io_utils import image_open_filter, image_save_filter, load_annotation, load_raster_image, save_annotation
from .models import Annotation


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MagicBorder")
        self.resize(1400, 920)

        self.canvas = ImageCanvas(self)
        self.setCentralWidget(self.canvas)

        self._current_annotation_path: Path | None = None

        self._create_actions()
        self._apply_action_visuals()
        self._create_menus()
        self._create_toolbar()

        self.statusBar().showMessage("Откройте фотографию листа растения.")
        self.canvas.message_changed.connect(self.statusBar().showMessage)
        self.canvas.image_state_changed.connect(self._update_action_states)
        self.canvas.contour_state_changed.connect(self._update_action_states)
        self._update_action_states()

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

        initial_name = "annotation.json"
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
        if reset_annotation:
            self._current_annotation_path = None
        return True

    def _build_annotation(self, annotation_path: Path) -> Annotation:
        image_path = self.canvas.current_image_path()
        image_size = self.canvas.image_size()
        if image_path is None or image_size is None:
            raise ValueError("Нет изображения для создания аннотации.")

        try:
            image_reference = os.path.relpath(image_path, annotation_path.parent)
        except ValueError:
            image_reference = str(image_path)

        width, height = image_size
        return Annotation(
            image_path=image_reference,
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

        stored_path = Path(annotation.image_path)
        candidates: list[Path] = []
        if stored_path.is_absolute():
            candidates.append(stored_path)
        else:
            candidates.append((annotation_path.parent / stored_path).resolve())
            candidates.append((annotation_path.parent / stored_path.name).resolve())

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

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
