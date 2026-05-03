from __future__ import annotations

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

from .main_window import MainWindow


def apply_light_theme(app: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#f5f7fb"))
    palette.setColor(QPalette.WindowText, QColor("#1f2937"))
    palette.setColor(QPalette.Base, QColor("#ffffff"))
    palette.setColor(QPalette.AlternateBase, QColor("#f1f4f8"))
    palette.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipText, QColor("#1f2937"))
    palette.setColor(QPalette.Text, QColor("#1f2937"))
    palette.setColor(QPalette.Button, QColor("#eef2f7"))
    palette.setColor(QPalette.ButtonText, QColor("#1f2937"))
    palette.setColor(QPalette.BrightText, QColor("#b42318"))
    palette.setColor(QPalette.Link, QColor("#176b87"))
    palette.setColor(QPalette.Highlight, QColor("#cdeaf2"))
    palette.setColor(QPalette.HighlightedText, QColor("#102a32"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#8a94a6"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#8a94a6"))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#8a94a6"))
    app.setPalette(palette)
    app.setStyleSheet(
        "QToolTip { color: #1f2937; background: #ffffff; border: 1px solid #ccd6e1; padding: 4px; }"
        "QMenuBar, QMenu { background: #ffffff; color: #1f2937; }"
        "QMenu::item:selected { background: #e6f4f7; color: #102a32; }"
        "QStatusBar { background: #f5f7fb; color: #475467; }"
    )


def main() -> int:
    QApplication.setAttribute(Qt.AA_DontShowIconsInMenus, False)
    app = QApplication(sys.argv)
    app.setApplicationName("MagicBorder")
    app.setOrganizationName("MagicBorder")
    app.setStyle("Fusion")
    apply_light_theme(app)

    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
