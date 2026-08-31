from __future__ import annotations

import unittest

from magicborder.oiv import default_oiv_catalog


class OivClassifierTest(unittest.TestCase):
    def test_catalog_contains_requested_codes_without_oiv_616(self) -> None:
        catalog = default_oiv_catalog()

        codes = {trait.code for trait in catalog.traits}
        self.assertEqual(
            codes,
            {
                "OIV 601",
                "OIV 602",
                "OIV 603",
                "OIV 604",
                "OIV 605",
                "OIV 606",
                "OIV 607",
                "OIV 608",
                "OIV 609",
                "OIV 610",
                "OIV 611",
                "OIV 612",
                "OIV 613",
                "OIV 614",
                "OIV 615",
                "OIV 617",
            },
        )
        self.assertNotIn("OIV 616", codes)

        self.assertEqual(
            [trait.code for trait in catalog.traits_for_tool("angle")],
            ["OIV 607", "OIV 608", "OIV 609", "OIV 610"],
        )
        self.assertEqual(
            [trait.code for trait in catalog.traits_for_tool("segment")],
            [
                "OIV 601",
                "OIV 602",
                "OIV 603",
                "OIV 604",
                "OIV 605",
                "OIV 606",
                "OIV 611",
                "OIV 612",
                "OIV 613",
                "OIV 614",
                "OIV 615",
                "OIV 617",
            ],
        )

    def test_classifies_lengths_by_midpoint_boundaries(self) -> None:
        catalog = default_oiv_catalog()

        self.assertEqual(
            catalog.classify("OIV 601", 89.9, tool_kind="segment").score, 1
        )
        self.assertEqual(catalog.classify("OIV 601", 90, tool_kind="segment").score, 3)
        self.assertEqual(catalog.classify("OIV 601", 136, tool_kind="segment").score, 5)
        self.assertEqual(catalog.classify("OIV 601", 180, tool_kind="segment").score, 9)

    def test_classifies_angles_and_reports_incompatible_tool(self) -> None:
        catalog = default_oiv_catalog()

        self.assertEqual(catalog.classify("OIV 607", 29.9, tool_kind="angle").score, 1)
        self.assertEqual(catalog.classify("OIV 607", 30, tool_kind="angle").score, 3)
        self.assertEqual(catalog.classify("OIV 607", 55.5, tool_kind="angle").score, 5)
        self.assertEqual(catalog.classify("OIV 607", 70, tool_kind="angle").score, 9)

        incompatible = catalog.classify("OIV 607", 40, tool_kind="segment")
        self.assertFalse(incompatible.ok)
        self.assertEqual(incompatible.status, "incompatible_tool")

        unknown = catalog.classify("OIV 999", 40, tool_kind="angle")
        self.assertFalse(unknown.ok)
        self.assertEqual(unknown.status, "unknown_trait")
