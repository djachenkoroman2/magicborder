from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from magicborder.icons import ACTION_VISUALS, ICON_DIR

MEASUREMENT_ACTION_ICONS = {
    "calibrate_scale": "calibrate-scale",
    "reset_calibration": "reset-calibration",
    "measure_angle": "measure-angle",
    "delete_angle": "delete-angle",
    "measure_segment": "measure-segment",
    "delete_segment": "delete-segment",
}
CANVAS_VISIBILITY_ACTION_ICONS = {
    "show_all_canvas_elements": "show-all-canvas-elements",
    "hide_all_canvas_elements": "hide-all-canvas-elements",
}


class ActionIconResourcesTest(unittest.TestCase):
    def test_all_action_visuals_reference_existing_svg_files(self) -> None:
        missing_icons = [
            visual.icon_name
            for visual in ACTION_VISUALS.values()
            if not (ICON_DIR / f"{visual.icon_name}.svg").is_file()
        ]

        self.assertEqual(missing_icons, [])

    def test_measurement_actions_use_dedicated_icons(self) -> None:
        for action_name, icon_name in MEASUREMENT_ACTION_ICONS.items():
            self.assertEqual(ACTION_VISUALS[action_name].icon_name, icon_name)

        self.assertEqual(
            len(set(MEASUREMENT_ACTION_ICONS.values())),
            len(MEASUREMENT_ACTION_ICONS),
        )
        for action_name in MEASUREMENT_ACTION_ICONS:
            self.assertNotIn(
                ACTION_VISUALS[action_name].icon_name,
                {"actual-size", "delete-contour"},
            )

    def test_measurement_icon_files_are_parseable_svgs(self) -> None:
        for icon_name in MEASUREMENT_ACTION_ICONS.values():
            root = ET.parse(ICON_DIR / f"{icon_name}.svg").getroot()

            self.assertEqual(root.tag.rsplit("}", maxsplit=1)[-1], "svg")
            self.assertEqual(root.attrib["width"], "32")
            self.assertEqual(root.attrib["height"], "32")
            self.assertEqual(root.attrib["viewBox"], "0 0 32 32")

    def test_canvas_visibility_actions_use_dedicated_icons(self) -> None:
        for action_name, icon_name in CANVAS_VISIBILITY_ACTION_ICONS.items():
            self.assertEqual(ACTION_VISUALS[action_name].icon_name, icon_name)

        self.assertEqual(
            len(set(CANVAS_VISIBILITY_ACTION_ICONS.values())),
            len(CANVAS_VISIBILITY_ACTION_ICONS),
        )
        for action_name in CANVAS_VISIBILITY_ACTION_ICONS:
            self.assertNotIn(
                ACTION_VISUALS[action_name].icon_name,
                {"default-view", "measure-angle", "measure-segment"},
            )

    def test_canvas_visibility_icon_files_are_parseable_svgs(self) -> None:
        for icon_name in CANVAS_VISIBILITY_ACTION_ICONS.values():
            root = ET.parse(ICON_DIR / f"{icon_name}.svg").getroot()

            self.assertEqual(root.tag.rsplit("}", maxsplit=1)[-1], "svg")
            self.assertEqual(root.attrib["width"], "32")
            self.assertEqual(root.attrib["height"], "32")
            self.assertEqual(root.attrib["viewBox"], "0 0 32 32")
