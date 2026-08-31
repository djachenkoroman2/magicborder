from __future__ import annotations

from typing import Any

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette

from magicborder import main as main_module
from magicborder.main import apply_light_theme, main

EXPECTED_ACTIVE_COLORS = {
    QPalette.Window: "#f5f7fb",
    QPalette.WindowText: "#1f2937",
    QPalette.Base: "#ffffff",
    QPalette.AlternateBase: "#f1f4f8",
    QPalette.ToolTipBase: "#ffffff",
    QPalette.ToolTipText: "#1f2937",
    QPalette.Text: "#1f2937",
    QPalette.Button: "#eef2f7",
    QPalette.ButtonText: "#1f2937",
    QPalette.BrightText: "#b42318",
    QPalette.Link: "#176b87",
    QPalette.Highlight: "#cdeaf2",
    QPalette.HighlightedText: "#102a32",
}
EXPECTED_DISABLED_COLORS = {
    QPalette.Text: "#8a94a6",
    QPalette.ButtonText: "#8a94a6",
    QPalette.WindowText: "#8a94a6",
}


class _ThemeRecorder:
    """Минимальный дублёр QApplication для проверки apply_light_theme."""

    def __init__(self) -> None:
        self.palette: QPalette | None = None
        self.style_sheet: str = ""

    def setPalette(self, palette: QPalette) -> None:  # noqa: N802 - Qt API
        self.palette = palette

    def setStyleSheet(self, style_sheet: str) -> None:  # noqa: N802 - Qt API
        self.style_sheet = style_sheet


class TestApplyLightTheme:
    def test_active_palette_roles(self) -> None:
        recorder = _ThemeRecorder()

        apply_light_theme(recorder)

        assert recorder.palette is not None
        for role, expected in EXPECTED_ACTIVE_COLORS.items():
            assert recorder.palette.color(QPalette.Active, role).name() == expected

    def test_disabled_palette_roles(self) -> None:
        recorder = _ThemeRecorder()

        apply_light_theme(recorder)

        assert recorder.palette is not None
        for role, expected in EXPECTED_DISABLED_COLORS.items():
            assert recorder.palette.color(QPalette.Disabled, role).name() == expected

    def test_stylesheet_covers_expected_widgets(self) -> None:
        recorder = _ThemeRecorder()

        apply_light_theme(recorder)

        for selector in ("QToolTip", "QMenuBar, QMenu", "QMenu::item:selected", "QStatusBar"):
            assert selector in recorder.style_sheet


class _FakeApplication:
    created: list[_FakeApplication] = []
    attributes: list[tuple[Any, bool]] = []

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.application_name = ""
        self.organization_name = ""
        self.style = ""
        self.palette: QPalette | None = None
        self.style_sheet = ""
        self.exec_calls = 0
        _FakeApplication.created.append(self)

    @classmethod
    def setAttribute(cls, attribute: Any, value: bool) -> None:  # noqa: N802 - Qt API
        cls.attributes.append((attribute, value))

    def setApplicationName(self, name: str) -> None:  # noqa: N802 - Qt API
        self.application_name = name

    def setOrganizationName(self, name: str) -> None:  # noqa: N802 - Qt API
        self.organization_name = name

    def setStyle(self, style: str) -> None:  # noqa: N802 - Qt API
        self.style = style

    def setPalette(self, palette: QPalette) -> None:  # noqa: N802 - Qt API
        self.palette = palette

    def setStyleSheet(self, style_sheet: str) -> None:  # noqa: N802 - Qt API
        self.style_sheet = style_sheet

    def exec_(self) -> int:
        self.exec_calls += 1
        return 7


class _FakeMainWindow:
    created: list[_FakeMainWindow] = []

    def __init__(self) -> None:
        self.shown = False
        _FakeMainWindow.created.append(self)

    def show(self) -> None:
        self.shown = True


@pytest.fixture()
def fake_application(monkeypatch: pytest.MonkeyPatch) -> type[_FakeApplication]:
    _FakeApplication.created.clear()
    _FakeApplication.attributes.clear()
    _FakeMainWindow.created.clear()
    monkeypatch.setattr(main_module, "QApplication", _FakeApplication)
    monkeypatch.setattr(main_module, "MainWindow", _FakeMainWindow)
    monkeypatch.setattr(main_module.sys, "argv", ["magicborder"])
    return _FakeApplication


class TestMain:
    def test_application_is_configured(self, fake_application: type[_FakeApplication]) -> None:
        exit_code = main()

        assert exit_code == 7
        app = fake_application.created[0]
        assert app.argv == ["magicborder"]
        assert app.application_name == "MagicBorder"
        assert app.organization_name == "MagicBorder"
        assert app.style == "Fusion"
        assert app.exec_calls == 1

    def test_icons_in_menus_attribute_is_enabled(
        self,
        fake_application: type[_FakeApplication],
    ) -> None:
        main()

        assert (Qt.AA_DontShowIconsInMenus, False) in fake_application.attributes

    def test_light_theme_is_applied(self, fake_application: type[_FakeApplication]) -> None:
        main()

        app = fake_application.created[0]
        assert app.palette is not None
        assert app.palette.color(QPalette.Window).name() == "#f5f7fb"
        assert "QStatusBar" in app.style_sheet

    def test_main_window_is_created_and_shown(
        self,
        fake_application: type[_FakeApplication],
    ) -> None:
        main()

        assert len(_FakeMainWindow.created) == 1
        assert _FakeMainWindow.created[0].shown is True
