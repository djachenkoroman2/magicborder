from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from magicborder.path_utils import annotation_image_candidates, portable_path_reference


class PathUtilsTest(unittest.TestCase):
    def test_portable_path_reference_uses_forward_slashes(self) -> None:
        base_dir = Path("annotations")
        image_path = base_dir / "images" / "leaf.png"

        self.assertEqual(
            portable_path_reference(image_path, base_dir),
            "images/leaf.png",
        )

    def test_portable_path_reference_handles_windows_drive_mismatch(self) -> None:
        image_path = Path("C:/data/leaf.png")

        with patch("magicborder.path_utils.os.path.relpath", side_effect=ValueError):
            reference = portable_path_reference(image_path, Path("D:/annotations"))

        self.assertEqual(reference, image_path.as_posix())

    def test_annotation_image_candidates_accept_windows_separators(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            annotation_path = Path(temp_dir) / "annotation.json"

            candidates = annotation_image_candidates("images\\leaf.png", annotation_path)

            self.assertIn((annotation_path.parent / "images" / "leaf.png").resolve(), candidates)
            self.assertIn((annotation_path.parent / "leaf.png").resolve(), candidates)

    def test_annotation_image_candidates_ignore_blank_references(self) -> None:
        self.assertEqual(annotation_image_candidates("   ", Path("annotation.json")), [])


if __name__ == "__main__":
    unittest.main()
