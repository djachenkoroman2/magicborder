from __future__ import annotations

import ast
import inspect
import textwrap
import xml.etree.ElementTree as ET

import pytest
from PyQt5.QtWidgets import QAction

from magicborder.icons import (
    ACTION_VISUALS,
    ICON_DIR,
    TOOLBAR_ICON_SIZE,
    ActionVisual,
    apply_action_visual,
    load_icon,
)

# Псевдоним команды: визуал объявлен, но применяется через export_project_excel.
ALIAS_ACTION_VISUALS = {"export_project_csv"}


def _applied_visual_keys() -> set[str]:
    from magicborder.main_window import MainWindow

    source = inspect.getsource(MainWindow._apply_action_visuals)
    tree = ast.parse(textwrap.dedent(source))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys.update(
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
    return keys


class TestApplyActionVisual:
    def test_all_visual_fields_are_written_to_the_action(self, qapp) -> None:
        action = QAction("Тест")
        visual = ActionVisual(
            icon_name="about",
            tool_tip="О программе",
            status_tip="Показать информацию о приложении.",
        )

        apply_action_visual(action, visual)

        assert action.icon().isNull() is False
        assert action.toolTip() == "О программе"
        assert action.statusTip() == "Показать информацию о приложении."
        assert action.whatsThis() == "Показать информацию о приложении."
        assert action.isIconVisibleInMenu() is True

    def test_every_declared_visual_can_be_applied(self, qapp) -> None:
        for name, visual in ACTION_VISUALS.items():
            action = QAction(name)

            apply_action_visual(action, visual)

            assert action.icon().isNull() is False, name
            assert action.toolTip() == visual.tool_tip
            assert action.statusTip() == visual.status_tip


class TestActionVisualsConsistency:
    def test_every_applied_action_has_a_visual(self, qapp) -> None:
        missing = _applied_visual_keys() - set(ACTION_VISUALS)

        assert missing == set()

    def test_only_the_documented_alias_is_unused(self, qapp) -> None:
        orphaned = set(ACTION_VISUALS) - _applied_visual_keys()

        assert orphaned == ALIAS_ACTION_VISUALS

    def test_alias_visual_matches_its_target(self) -> None:
        assert (
            ACTION_VISUALS["export_project_csv"]
            == ACTION_VISUALS["export_project_excel"]
        )

    def test_window_actions_receive_their_visuals(self, qapp) -> None:
        from magicborder.main_window import MainWindow

        window = MainWindow()
        try:
            actions = {
                name: getattr(window, name)
                for name in dir(window)
                if name.endswith("_action")
                and isinstance(getattr(window, name), QAction)
            }

            assert actions
            for name, action in actions.items():
                assert action.icon().isNull() is False, name
                assert action.statusTip(), name
                assert action.whatsThis() == action.statusTip(), name
        finally:
            window.close()
            window.deleteLater()


class TestIconResources:
    def test_every_svg_in_icon_dir_parses(self) -> None:
        svg_paths = sorted(ICON_DIR.glob("*.svg"))

        assert svg_paths
        for svg_path in svg_paths:
            root = ET.parse(svg_path).getroot()

            assert root.tag.rsplit("}", maxsplit=1)[-1] == "svg", svg_path.name

    def test_every_svg_uses_the_shared_viewbox(self) -> None:
        for svg_path in sorted(ICON_DIR.glob("*.svg")):
            root = ET.parse(svg_path).getroot()

            assert root.attrib.get("viewBox") == "0 0 32 32", svg_path.name

    def test_load_icon_returns_empty_icon_for_unknown_name(self, qapp) -> None:
        assert load_icon("такой-иконки-нет").isNull() is True

    def test_load_icon_returns_real_icon_for_known_name(self, qapp) -> None:
        assert load_icon("about").isNull() is False

    def test_toolbar_icon_size(self) -> None:
        assert (TOOLBAR_ICON_SIZE.width(), TOOLBAR_ICON_SIZE.height()) == (30, 30)


@pytest.mark.parametrize("action_name", sorted(ACTION_VISUALS))
def test_visual_texts_are_not_empty(action_name: str) -> None:
    visual = ACTION_VISUALS[action_name]

    assert visual.icon_name.strip()
    assert visual.tool_tip.strip()
    assert visual.status_tip.strip()
