from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)


class PropertyBrowser(QTreeWidget):
    """Small PyQt5 property-browser widget used when QtPropertyBrowser is unavailable."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("propertyBrowser")
        self.setColumnCount(2)
        self.setHeaderLabels(["Свойство", "Значение"])
        self.setHeaderHidden(True)
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
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)

        self._groups: dict[str, QTreeWidgetItem] = {}
        self._properties: dict[str, QTreeWidgetItem] = {}

    def add_group(self, title: str, *, expanded: bool = False) -> QTreeWidgetItem:
        group_item = QTreeWidgetItem([title, ""])
        group_item.setData(0, Qt.UserRole, "group")
        group_item.setFirstColumnSpanned(True)
        group_item.setFlags(group_item.flags() & ~Qt.ItemIsEditable)

        font = QFont(group_item.font(0))
        font.setBold(True)
        group_item.setFont(0, font)

        self.addTopLevelItem(group_item)
        group_item.setExpanded(expanded)
        self._groups[title] = group_item
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
        group_item.addChild(property_item)
        self.setItemWidget(property_item, 1, editor)

        if editor.minimumHeight() > 0:
            row_height = max(24, editor.minimumSizeHint().height(), editor.minimumHeight())
        else:
            row_height = max(24, editor.minimumSizeHint().height(), editor.sizeHint().height())
        size_hint = QSize(120, row_height + 4)
        property_item.setSizeHint(0, size_hint)
        property_item.setSizeHint(1, size_hint)

        self._properties[key or label] = property_item
        return property_item

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
            group_item = self.topLevelItem(group_index)
            rows.append(f"--- {group_item.text(0)}")
            for child_index in range(group_item.childCount()):
                rows.append(group_item.child(child_index).text(0))
        return rows

    def is_property_visible(self, key_or_label: str) -> bool:
        item = self.property_item(key_or_label)
        parent = item.parent()
        while parent is not None:
            if not parent.isExpanded():
                return False
            parent = parent.parent()
        return not item.isHidden()
