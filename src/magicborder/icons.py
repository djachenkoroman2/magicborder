from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction

ICON_DIR = Path(__file__).resolve().parent / "assets" / "icons"
TOOLBAR_ICON_SIZE = QSize(30, 30)


@dataclass(frozen=True, slots=True)
class ActionVisual:
    icon_name: str
    tool_tip: str
    status_tip: str


ACTION_VISUALS: dict[str, ActionVisual] = {
    "open_image": ActionVisual(
        icon_name="open-image",
        tool_tip="Открыть изображение",
        status_tip="Открыть фотографию листа растения из файла.",
    ),
    "save_image": ActionVisual(
        icon_name="save-image",
        tool_tip="Сохранить изображение",
        status_tip="Сохранить текущее изображение без контура в файл.",
    ),
    "exit": ActionVisual(
        icon_name="exit-app",
        tool_tip="Выход",
        status_tip="Закрыть приложение MagicBorder.",
    ),
    "zoom_in": ActionVisual(
        icon_name="zoom-in",
        tool_tip="Увеличить",
        status_tip="Увеличить масштаб изображения на канвасе.",
    ),
    "zoom_out": ActionVisual(
        icon_name="zoom-out",
        tool_tip="Уменьшить",
        status_tip="Уменьшить масштаб изображения на канвасе.",
    ),
    "fit_image": ActionVisual(
        icon_name="fit-image",
        tool_tip="Показать целиком",
        status_tip="Вписать изображение целиком в область просмотра.",
    ),
    "actual_size": ActionVisual(
        icon_name="actual-size",
        tool_tip="Масштаб 100%",
        status_tip="Вернуть отображение изображения к масштабу 100%.",
    ),
    "default_view": ActionVisual(
        icon_name="default-view",
        tool_tip="Вид по умолчанию",
        status_tip="Вернуть стандартные пропорции панелей и вписать изображение в область просмотра.",
    ),
    "detect_contour": ActionVisual(
        icon_name="detect-contour",
        tool_tip="Определить контур",
        status_tip="Автоматически определить границу листа и построить контур.",
    ),
    "flatten_background": ActionVisual(
        icon_name="flatten-background",
        tool_tip="Выровнять фон",
        status_tip="Сделать область вне текущего контура белой.",
    ),
    "save_annotation": ActionVisual(
        icon_name="save-annotation",
        tool_tip="Сохранить аннотацию",
        status_tip="Сохранить текущий контур в JSON-файл аннотации.",
    ),
    "open_annotation": ActionVisual(
        icon_name="open-annotation",
        tool_tip="Открыть аннотацию",
        status_tip="Загрузить контур из JSON-файла аннотации.",
    ),
    "about": ActionVisual(
        icon_name="about",
        tool_tip="О программе",
        status_tip="Показать информацию о приложении MagicBorder.",
    ),
}


def load_icon(icon_name: str) -> QIcon:
    return QIcon(str(ICON_DIR / f"{icon_name}.svg"))


def apply_action_visual(action: QAction, visual: ActionVisual) -> None:
    action.setIcon(load_icon(visual.icon_name))
    action.setToolTip(visual.tool_tip)
    action.setStatusTip(visual.status_tip)
    action.setWhatsThis(visual.status_tip)
    action.setIconVisibleInMenu(True)
