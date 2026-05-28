from __future__ import annotations

from PyQt5.QtCore import QEvent, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)


PROPERTY_KEY_DEFAULT_WIDTH = 150
PROPERTY_KEY_MIN_WIDTH = 72
PROPERTY_VALUE_MIN_WIDTH = 90
PROPERTY_ROW_MIN_HEIGHT = 24
PROPERTY_ROW_VERTICAL_PADDING = 8


class PropertyValueLabel(QLabel):
    text_changed = pyqtSignal()

    def __init__(self, text: str = "-", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._sync_tooltip()

    def setText(self, text: str) -> None:
        super().setText(text)
        self._sync_tooltip()
        self.updateGeometry()
        self.text_changed.emit()

    def _sync_tooltip(self) -> None:
        text = self.text()
        self.setToolTip(text if text and text != "-" else "")


class PropertyBrowser(QTreeWidget):
    """Small PyQt5 property-browser widget used when QtPropertyBrowser is unavailable."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("propertyBrowser")
        self.setColumnCount(2)
        self.setHeaderLabels(["Свойство", "Значение"])
        self.setHeaderHidden(False)
        self.setRootIsDecorated(True)
        self.setAnimated(False)
        self.setIndentation(14)
        self.setUniformRowHeights(False)
        self.setAlternatingRowColors(False)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        header = self.header()
        header.setStretchLastSection(True)
        header.setSectionsMovable(False)
        header.setSectionsClickable(False)
        header.setMinimumSectionSize(PROPERTY_KEY_MIN_WIDTH)
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.setColumnWidth(0, PROPERTY_KEY_DEFAULT_WIDTH)

        self._groups: dict[str, QTreeWidgetItem] = {}
        self._properties: dict[str, QTreeWidgetItem] = {}
        self._property_rows: list[tuple[QTreeWidgetItem, QWidget]] = []
        self._refresh_layout_pending = False
        self._clamping_key_column = False
        header.sectionResized.connect(self._handle_section_resized)

    def key_column_width(self) -> int:
        return self.columnWidth(0)

    def set_key_column_width(self, width: int) -> None:
        self.setColumnWidth(0, self._clamped_key_column_width(width))
        self.refresh_layout()

    def refresh_layout(self) -> None:
        self._refresh_layout_pending = False
        for item, editor in self._property_rows:
            row_height = self._property_row_height(item, editor)
            size_hint = QSize(120, row_height)
            item.setSizeHint(0, size_hint)
            item.setSizeHint(1, size_hint)

    def add_group(
        self,
        title: str,
        *,
        expanded: bool = False,
        parent: QTreeWidgetItem | None = None,
        key: str | None = None,
    ) -> QTreeWidgetItem:
        group_item = QTreeWidgetItem([title, ""])
        group_item.setData(0, Qt.UserRole, "group")
        group_item.setData(0, Qt.UserRole + 1, key or title)
        group_item.setFirstColumnSpanned(True)
        group_item.setFlags(group_item.flags() & ~Qt.ItemIsEditable)

        font = QFont(group_item.font(0))
        font.setBold(True)
        group_item.setFont(0, font)

        if parent is None:
            self.addTopLevelItem(group_item)
        else:
            parent.addChild(group_item)
        group_item.setExpanded(expanded)
        self._groups[key or title] = group_item
        return group_item

    def add_property(
        self,
        group_title: str,
        label: str,
        editor: QWidget,
        *,
        key: str | None = None,
    ) -> QTreeWidgetItem:
        group_item = self._groups[group_title]
        property_item = QTreeWidgetItem([label, ""])
        property_item.setData(0, Qt.UserRole, "property")
        property_item.setData(0, Qt.UserRole + 1, key or label)
        property_item.setFlags(property_item.flags() & ~Qt.ItemIsEditable)
        property_item.setToolTip(0, label)
        group_item.addChild(property_item)
        self._configure_editor(editor)
        self.setItemWidget(property_item, 1, editor)
        self._property_rows.append((property_item, editor))

        row_height = self._property_row_height(property_item, editor)
        size_hint = QSize(120, row_height)
        property_item.setSizeHint(0, size_hint)
        property_item.setSizeHint(1, size_hint)

        self._properties[key or label] = property_item
        return property_item

    def add_property_to_item(
        self,
        parent: QTreeWidgetItem,
        label: str,
        editor: QWidget,
        *,
        key: str | None = None,
    ) -> QTreeWidgetItem:
        property_item = QTreeWidgetItem([label, ""])
        property_item.setData(0, Qt.UserRole, "property")
        property_item.setData(0, Qt.UserRole + 1, key or label)
        property_item.setFlags(property_item.flags() & ~Qt.ItemIsEditable)
        property_item.setToolTip(0, label)
        parent.addChild(property_item)
        self._configure_editor(editor)
        self.setItemWidget(property_item, 1, editor)
        self._property_rows.append((property_item, editor))

        row_height = self._property_row_height(property_item, editor)
        size_hint = QSize(120, row_height)
        property_item.setSizeHint(0, size_hint)
        property_item.setSizeHint(1, size_hint)

        self._properties[key or label] = property_item
        return property_item

    def clear_children(self, parent: QTreeWidgetItem) -> None:
        while parent.childCount():
            child = parent.takeChild(0)
            self._forget_item(child)
        self.refresh_layout()

    def group_item(self, title: str) -> QTreeWidgetItem:
        try:
            return self._groups[title]
        except KeyError as exc:
            raise KeyError(f"Группа свойств не найдена: {title}") from exc

    def property_item(self, key_or_label: str) -> QTreeWidgetItem:
        try:
            return self._properties[key_or_label]
        except KeyError as exc:
            raise KeyError(f"Свойство не найдено: {key_or_label}") from exc

    def rows(self) -> list[str]:
        rows: list[str] = []
        for group_index in range(self.topLevelItemCount()):
            self._append_item_rows(self.topLevelItem(group_index), rows)
        return rows

    def is_property_visible(self, key_or_label: str) -> bool:
        item = self.property_item(key_or_label)
        parent = item.parent()
        while parent is not None:
            if not parent.isExpanded():
                return False
            parent = parent.parent()
        return not item.isHidden()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._clamp_current_key_column_width()
        self._schedule_refresh_layout()

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if event.type() in (
            QEvent.FontChange,
            QEvent.LayoutRequest,
            QEvent.Resize,
            QEvent.Show,
            QEvent.StyleChange,
        ):
            self._schedule_refresh_layout()
        return super().eventFilter(watched, event)

    def _configure_editor(self, editor: QWidget) -> None:
        editor.setMinimumWidth(0)
        editor.installEventFilter(self)
        if isinstance(editor, QLabel):
            editor.setWordWrap(True)
            editor.setMinimumWidth(0)
            editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            if editor.text() and editor.text() != "-":
                editor.setToolTip(editor.text())
        if isinstance(editor, PropertyValueLabel):
            editor.text_changed.connect(self._schedule_refresh_layout)

    def _append_item_rows(self, item: QTreeWidgetItem, rows: list[str]) -> None:
        if item.data(0, Qt.UserRole) == "group":
            rows.append(f"--- {item.text(0)}")
        else:
            rows.append(item.text(0))
        for child_index in range(item.childCount()):
            self._append_item_rows(item.child(child_index), rows)

    def _forget_item(self, item: QTreeWidgetItem) -> None:
        for child_index in reversed(range(item.childCount())):
            child = item.takeChild(child_index)
            self._forget_item(child)

        key = item.data(0, Qt.UserRole + 1)
        role = item.data(0, Qt.UserRole)
        if role == "group":
            for group_key, group_item in list(self._groups.items()):
                if group_item is item or group_key == key:
                    self._groups.pop(group_key, None)
        elif role == "property":
            for property_key, property_item in list(self._properties.items()):
                if property_item is item or property_key == key:
                    self._properties.pop(property_key, None)

        editor = self.itemWidget(item, 1)
        if editor is not None:
            self.removeItemWidget(item, 1)
            editor.deleteLater()
        self._property_rows = [
            (property_item, property_editor)
            for property_item, property_editor in self._property_rows
            if property_item is not item
        ]

    def _handle_section_resized(self, index: int, _old_size: int, _new_size: int) -> None:
        if index == 0:
            self._clamp_current_key_column_width()
        self._schedule_refresh_layout()

    def _schedule_refresh_layout(self) -> None:
        if self._refresh_layout_pending:
            return
        self._refresh_layout_pending = True
        QTimer.singleShot(0, self.refresh_layout)

    def _property_row_height(self, item: QTreeWidgetItem, editor: QWidget) -> int:
        editor_height = max(
            PROPERTY_ROW_MIN_HEIGHT,
            editor.minimumSizeHint().height(),
            editor.sizeHint().height(),
            editor.minimumHeight(),
        )
        if editor.hasHeightForWidth():
            value_width = max(PROPERTY_VALUE_MIN_WIDTH, self.columnWidth(1) - 8)
            editor_height = max(editor_height, editor.heightForWidth(value_width))

        key_width = max(1, self.columnWidth(0) - self.indentation() - 8)
        key_rect = QFontMetrics(item.font(0)).boundingRect(
            0,
            0,
            key_width,
            10_000,
            int(Qt.TextWordWrap),
            item.text(0),
        )
        key_height = max(PROPERTY_ROW_MIN_HEIGHT, key_rect.height())
        return max(editor_height, key_height) + PROPERTY_ROW_VERTICAL_PADDING

    def _clamped_key_column_width(self, width: int) -> int:
        width = max(PROPERTY_KEY_MIN_WIDTH, int(width))
        viewport_width = self.viewport().width()
        if viewport_width <= PROPERTY_KEY_MIN_WIDTH + PROPERTY_VALUE_MIN_WIDTH:
            return width
        return min(width, viewport_width - PROPERTY_VALUE_MIN_WIDTH)

    def _clamp_current_key_column_width(self) -> None:
        if self._clamping_key_column:
            return
        current_width = self.columnWidth(0)
        clamped_width = self._clamped_key_column_width(current_width)
        if clamped_width == current_width:
            return
        self._clamping_key_column = True
        try:
            self.setColumnWidth(0, clamped_width)
        finally:
            self._clamping_key_column = False
