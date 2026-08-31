from __future__ import annotations

import pytest
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QLabel, QLineEdit

from magicborder.property_browser import (
    PROPERTY_KEY_MIN_WIDTH,
    PROPERTY_VALUE_MIN_WIDTH,
    PropertyBrowser,
    PropertyGridOverlay,
    PropertyValueLabel,
)


@pytest.fixture()
def browser(qapp) -> PropertyBrowser:  # noqa: ARG001
    widget = PropertyBrowser()
    widget.resize(400, 300)
    return widget


@pytest.fixture()
def populated(browser: PropertyBrowser) -> PropertyBrowser:
    browser.add_group("Файл", expanded=True, key="file")
    browser.add_property(
        "file", "Имя файла", PropertyValueLabel("leaf.png"), key="file.name"
    )
    browser.add_property("file", "Размер", PropertyValueLabel("40x30"), key="file.size")
    return browser


class TestGroupAndPropertyLookup:
    def test_group_is_found_by_key(self, populated: PropertyBrowser) -> None:
        assert populated.group_item("file").text(0) == "Файл"

    def test_group_added_without_key_is_found_by_title(
        self, browser: PropertyBrowser
    ) -> None:
        browser.add_group("Контур")

        assert browser.group_item("Контур").text(0) == "Контур"

    def test_missing_group_raises_key_error(self, browser: PropertyBrowser) -> None:
        with pytest.raises(KeyError, match="Группа свойств не найдена"):
            browser.group_item("нет-такой")

    def test_property_is_found_by_key(self, populated: PropertyBrowser) -> None:
        assert populated.property_item("file.name").text(0) == "Имя файла"

    def test_property_is_found_by_label_when_no_key(
        self, browser: PropertyBrowser
    ) -> None:
        browser.add_group("Файл", key="file")
        browser.add_property("file", "Имя файла", PropertyValueLabel("leaf.png"))

        assert browser.property_item("Имя файла").text(0) == "Имя файла"

    def test_property_aliases_point_to_the_same_item(
        self, browser: PropertyBrowser
    ) -> None:
        browser.add_group("Файл", key="file")
        item = browser.add_property(
            "file",
            "Имя файла",
            PropertyValueLabel("leaf.png"),
            key="file.name",
            aliases=("Имя", "file_name"),
        )

        assert browser.property_item("Имя") is item
        assert browser.property_item("file_name") is item

    def test_missing_property_raises_key_error(self, browser: PropertyBrowser) -> None:
        with pytest.raises(KeyError, match="Свойство не найдено"):
            browser.property_item("нет-такого")


class TestAddPropertyToItem:
    def test_nested_property_is_registered(self, populated: PropertyBrowser) -> None:
        parent = populated.group_item("file")
        nested_group = populated.add_group(
            "Вложенная", parent=parent, key="file.nested"
        )

        item = populated.add_property_to_item(
            nested_group,
            "Ключ",
            PropertyValueLabel("значение"),
            key="file.nested.key",
            aliases=("nested_key",),
        )

        assert populated.property_item("file.nested.key") is item
        assert populated.property_item("nested_key") is item
        assert item.parent() is nested_group
        assert populated.itemWidget(item, 1) is not None

    def test_nested_property_appears_in_rows(self, populated: PropertyBrowser) -> None:
        parent = populated.group_item("file")
        populated.add_property_to_item(
            parent, "Дополнительно", PropertyValueLabel("да")
        )

        assert populated.rows() == [
            "--- Файл",
            "Имя файла",
            "Размер",
            "Дополнительно",
        ]


class TestClearChildren:
    def test_children_are_removed_from_the_registry(
        self, populated: PropertyBrowser
    ) -> None:
        group = populated.group_item("file")

        populated.clear_children(group)

        assert group.childCount() == 0
        assert populated.rows() == ["--- Файл"]
        with pytest.raises(KeyError):
            populated.property_item("file.name")
        with pytest.raises(KeyError):
            populated.property_item("file.size")

    def test_nested_groups_are_forgotten_too(self, populated: PropertyBrowser) -> None:
        parent = populated.group_item("file")
        nested = populated.add_group("Вложенная", parent=parent, key="file.nested")
        populated.add_property_to_item(
            nested, "Ключ", PropertyValueLabel("v"), key="nested.key"
        )

        populated.clear_children(parent)

        with pytest.raises(KeyError):
            populated.group_item("file.nested")
        with pytest.raises(KeyError):
            populated.property_item("nested.key")
        assert populated.group_item("file") is parent

    def test_property_rows_registry_is_pruned(self, populated: PropertyBrowser) -> None:
        assert len(populated._property_rows) == 2

        populated.clear_children(populated.group_item("file"))

        assert populated._property_rows == []

    def test_aliases_are_removed_with_their_item(
        self, browser: PropertyBrowser
    ) -> None:
        browser.add_group("Файл", key="file")
        browser.add_property(
            "file",
            "Имя файла",
            PropertyValueLabel("leaf.png"),
            key="file.name",
            aliases=("Имя",),
        )

        browser.clear_children(browser.group_item("file"))

        with pytest.raises(KeyError):
            browser.property_item("Имя")


class TestIsPropertyVisible:
    def test_visible_inside_expanded_group(self, populated: PropertyBrowser) -> None:
        assert populated.is_property_visible("file.name") is True

    def test_hidden_inside_collapsed_group(self, populated: PropertyBrowser) -> None:
        populated.group_item("file").setExpanded(False)

        assert populated.is_property_visible("file.name") is False

    def test_hidden_inside_collapsed_ancestor(self, populated: PropertyBrowser) -> None:
        parent = populated.group_item("file")
        nested = populated.add_group(
            "Вложенная", expanded=True, parent=parent, key="file.nested"
        )
        populated.add_property_to_item(
            nested, "Ключ", PropertyValueLabel("v"), key="nested.key"
        )

        assert populated.is_property_visible("nested.key") is True

        parent.setExpanded(False)
        assert populated.is_property_visible("nested.key") is False

    def test_explicitly_hidden_item(self, populated: PropertyBrowser) -> None:
        populated.property_item("file.name").setHidden(True)

        assert populated.is_property_visible("file.name") is False


class TestEditorConfiguration:
    def test_labels_get_word_wrap_and_tooltip(self, browser: PropertyBrowser) -> None:
        browser.add_group("Файл", key="file")
        editor = QLabel("длинное значение свойства")

        browser.add_property("file", "Свойство", editor)

        assert editor.wordWrap() is True
        assert editor.minimumWidth() == 0
        assert editor.toolTip() == "длинное значение свойства"

    def test_placeholder_label_gets_no_tooltip(self, browser: PropertyBrowser) -> None:
        browser.add_group("Файл", key="file")
        editor = QLabel("-")

        browser.add_property("file", "Свойство", editor)

        assert editor.toolTip() == ""

    def test_non_label_editor_is_only_width_relaxed(
        self, browser: PropertyBrowser
    ) -> None:
        browser.add_group("Файл", key="file")
        editor = QLineEdit("текст")

        browser.add_property("file", "Свойство", editor)

        assert editor.minimumWidth() == 0
        assert editor.toolTip() == ""

    def test_property_value_label_text_change_schedules_refresh(
        self,
        populated: PropertyBrowser,
    ) -> None:
        populated.refresh_layout()
        assert populated._refresh_layout_pending is False
        editor = populated.itemWidget(populated.property_item("file.name"), 1)

        editor.setText("новое значение")

        assert populated._refresh_layout_pending is True

    @pytest.mark.parametrize(
        "event_type",
        [
            QEvent.FontChange,
            QEvent.LayoutRequest,
            QEvent.Resize,
            QEvent.Show,
            QEvent.StyleChange,
        ],
    )
    def test_event_filter_schedules_refresh(
        self,
        populated: PropertyBrowser,
        event_type: int,
    ) -> None:
        populated.refresh_layout()
        editor = populated.itemWidget(populated.property_item("file.name"), 1)

        populated.eventFilter(editor, QEvent(event_type))

        assert populated._refresh_layout_pending is True

    def test_event_filter_ignores_other_events(
        self, populated: PropertyBrowser
    ) -> None:
        populated.refresh_layout()
        editor = populated.itemWidget(populated.property_item("file.name"), 1)

        populated.eventFilter(editor, QEvent(QEvent.Paint))

        assert populated._refresh_layout_pending is False


class TestPropertyRowHeight:
    def test_multiline_value_is_taller_than_single_line(
        self, browser: PropertyBrowser
    ) -> None:
        browser.add_group("Файл", key="file")
        short_item = browser.add_property(
            "file", "Короткое", PropertyValueLabel("да"), key="a"
        )
        long_item = browser.add_property(
            "file",
            "Длинное",
            PropertyValueLabel(
                "очень длинное значение свойства, которое точно не влезет в одну строку виджета"
            ),
            key="b",
        )
        browser.setColumnWidth(0, 120)
        browser.refresh_layout()

        short_height = browser._property_row_height(
            short_item, browser.itemWidget(short_item, 1)
        )
        long_height = browser._property_row_height(
            long_item, browser.itemWidget(long_item, 1)
        )

        assert long_height > short_height

    def test_long_key_text_also_increases_height(
        self, browser: PropertyBrowser
    ) -> None:
        browser.add_group("Файл", key="file")
        item = browser.add_property(
            "file",
            "Очень длинное имя свойства, которое переносится на несколько строк",
            PropertyValueLabel("да"),
            key="a",
        )
        browser.setColumnWidth(0, PROPERTY_KEY_MIN_WIDTH)

        assert browser._property_row_height(item, browser.itemWidget(item, 1)) > 32


class TestKeyColumnWidthClamping:
    def test_width_below_minimum_is_raised(self, browser: PropertyBrowser) -> None:
        assert browser._clamped_key_column_width(10) == PROPERTY_KEY_MIN_WIDTH

    def test_width_above_maximum_is_lowered(self, browser: PropertyBrowser) -> None:
        viewport_width = browser.viewport().width()

        clamped = browser._clamped_key_column_width(viewport_width)

        assert clamped == viewport_width - PROPERTY_VALUE_MIN_WIDTH

    def test_narrow_viewport_keeps_requested_width(
        self, browser: PropertyBrowser
    ) -> None:
        browser.resize(PROPERTY_KEY_MIN_WIDTH + PROPERTY_VALUE_MIN_WIDTH - 20, 200)

        assert browser._clamped_key_column_width(200) == 200

    def test_set_key_column_width_applies_clamping(
        self, browser: PropertyBrowser
    ) -> None:
        browser.set_key_column_width(5)

        assert browser.key_column_width() == PROPERTY_KEY_MIN_WIDTH

    def test_resize_clamps_current_width(self, browser: PropertyBrowser) -> None:
        browser.setColumnWidth(0, 350)

        browser.resize(300, 300)

        assert (
            browser.key_column_width()
            <= browser.viewport().width() - PROPERTY_VALUE_MIN_WIDTH
        )

    def test_clamping_is_not_reentrant(self, browser: PropertyBrowser) -> None:
        browser._clamping_key_column = True
        browser.setColumnWidth(0, 5_000)

        browser._clamp_current_key_column_width()

        assert browser.columnWidth(0) == 5_000


class TestPropertyValueLabel:
    def test_tooltip_follows_text(self, qapp) -> None:
        label = PropertyValueLabel("значение")

        assert label.toolTip() == "значение"

        label.setText("другое значение")
        assert label.toolTip() == "другое значение"

    def test_placeholder_and_empty_text_have_no_tooltip(self, qapp) -> None:
        label = PropertyValueLabel("-")
        assert label.toolTip() == ""

        label.setText("")
        assert label.toolTip() == ""

    def test_text_changed_signal_is_emitted(self, qapp) -> None:
        label = PropertyValueLabel("значение")
        emitted: list[int] = []
        label.text_changed.connect(lambda: emitted.append(1))

        label.setText("новое")

        assert emitted == [1]

    def test_default_configuration(self, qapp) -> None:
        label = PropertyValueLabel()

        assert label.wordWrap() is True
        assert label.minimumWidth() == 0
        assert label.textInteractionFlags() & Qt.TextSelectableByMouse


class TestPropertyGridOverlay:
    def test_overlay_is_attached_to_the_viewport(
        self, browser: PropertyBrowser
    ) -> None:
        overlay = browser._grid_overlay

        assert isinstance(overlay, PropertyGridOverlay)
        assert overlay.parent() is browser.viewport()
        assert overlay.testAttribute(Qt.WA_TransparentForMouseEvents) is True

    def test_paint_event_on_empty_tree(self, browser: PropertyBrowser) -> None:
        assert not browser._grid_overlay.grab().isNull()

    def test_paint_event_with_items(self, populated: PropertyBrowser) -> None:
        populated.refresh_layout()

        assert not populated._grid_overlay.grab().isNull()

    def test_paint_event_on_scrolled_tree(self, browser: PropertyBrowser) -> None:
        browser.resize(300, 80)
        browser.add_group("Файл", expanded=True, key="file")
        for index in range(40):
            browser.add_property(
                "file",
                f"Свойство {index}",
                PropertyValueLabel(f"значение {index}"),
                key=f"file.{index}",
            )
        browser.refresh_layout()
        browser.verticalScrollBar().setValue(browser.verticalScrollBar().maximum())

        assert not browser._grid_overlay.grab().isNull()

    def test_paint_event_with_single_column(self, browser: PropertyBrowser) -> None:
        browser.setColumnCount(1)

        assert not browser._grid_overlay.grab().isNull()


def test_empty_area_click_signal(browser: PropertyBrowser, qapp) -> None:  # noqa: ARG001
    from PyQt5.QtCore import QPoint
    from PyQt5.QtGui import QMouseEvent

    emitted: list[int] = []
    browser.empty_area_clicked.connect(lambda: emitted.append(1))
    event = QMouseEvent(
        QEvent.MouseButtonPress,
        QPoint(10, browser.viewport().height() - 5),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )

    browser.mousePressEvent(event)

    assert emitted == [1]
