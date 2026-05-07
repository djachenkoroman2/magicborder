from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from magicborder.io_utils import load_project, save_project
from magicborder.models import PROJECT_FORMAT_VERSION, Annotation, Point, ProjectDocument, ProjectImageRecord


class ProjectModelsTest(unittest.TestCase):
    def test_project_round_trip_preserves_image_annotation(self) -> None:
        annotation = Annotation(
            image_path="images/leaf.png",
            image_width=10,
            image_height=8,
            points=[Point(1, 1), Point(8, 1), Point(6, 6)],
        )
        project = ProjectDocument(
            name="leaves",
            images=[
                ProjectImageRecord(
                    id="leaf-1",
                    relative_path="images/leaf.png",
                    display_name="leaf.png",
                    image_width=10,
                    image_height=8,
                    annotation=annotation,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "leaves.json"
            save_project(project_path, project)
            payload = json.loads(project_path.read_text(encoding="utf-8"))

            loaded_project = load_project(project_path)

        record_payload = payload["images"][0]
        self.assertEqual(payload["version"], PROJECT_FORMAT_VERSION)
        self.assertIn("file", record_payload)
        self.assertIn("contour", record_payload)
        self.assertIn("location", record_payload)
        self.assertIn("details", record_payload)
        self.assertNotIn("metadata", record_payload)
        self.assertNotIn("path", record_payload)
        self.assertEqual(record_payload["file"]["id"], "leaf-1")
        self.assertEqual(record_payload["file"]["path"], "images/leaf.png")
        self.assertIsNotNone(record_payload["contour"]["annotation"])
        self.assertEqual(loaded_project.name, "leaves")
        self.assertEqual(len(loaded_project.images), 1)
        loaded_record = loaded_project.images[0]
        self.assertEqual(loaded_record.relative_path, "images/leaf.png")
        self.assertIsNotNone(loaded_record.annotation)
        self.assertEqual(loaded_record.annotation.image_width, 10)
        self.assertEqual(len(loaded_record.annotation.points), 3)

    def test_project_keeps_corrupt_annotation_payload(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "version": PROJECT_FORMAT_VERSION,
                "name": "broken",
                "images": [
                    {
                        "file": {
                            "id": "leaf-1",
                            "path": "images/leaf.png",
                            "display_name": "leaf.png",
                        },
                        "contour": {"annotation": {"points": []}},
                        "location": {},
                        "details": {},
                    }
                ],
            }
        )

        record = project.images[0]

        self.assertIsNone(record.annotation)
        self.assertIsNotNone(record.annotation_error)
        self.assertEqual(record.to_dict()["contour"]["annotation"], {"points": []})

    def test_project_image_metadata_round_trip_uses_grouped_json(self) -> None:
        project = ProjectDocument(
            name="metadata",
            images=[
                ProjectImageRecord(
                    id="leaf-1",
                    relative_path="images/leaf.png",
                    display_name="leaf.png",
                    metadata={
                        "added_at": "2026-05-03T10:20:30+03:00",
                        "captured_at": "2026-05-04T08:00:00+03:00",
                        "humidity": "55",
                        "diagnosis": "leaf_blight",
                    },
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "metadata.json"
            save_project(project_path, project)
            payload = json.loads(project_path.read_text(encoding="utf-8"))

            loaded_project = load_project(project_path)

        record_payload = payload["images"][0]
        self.assertEqual(record_payload["file"]["added_at"], "2026-05-03T10:20:30+03:00")
        self.assertEqual(record_payload["file"]["captured_at"], "2026-05-04T08:00:00+03:00")
        self.assertEqual(record_payload["location"]["humidity"], "55")
        self.assertEqual(record_payload["details"]["diagnosis"], "leaf_blight")
        self.assertNotIn("metadata", record_payload)

        metadata = loaded_project.images[0].metadata
        self.assertEqual(metadata["added_at"], "2026-05-03T10:20:30+03:00")
        self.assertEqual(metadata["captured_at"], "2026-05-04T08:00:00+03:00")
        self.assertEqual(metadata["humidity"], "55")
        self.assertEqual(metadata["diagnosis"], "leaf_blight")

    def test_project_image_group_defaults(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "version": PROJECT_FORMAT_VERSION,
                "name": "group_defaults",
                "images": [
                    {
                        "file": {
                            "id": "leaf-1",
                            "path": "images/leaf.png",
                            "display_name": "leaf.png",
                        },
                        "contour": {},
                        "location": {},
                        "details": {},
                    }
                ],
            }
        )

        metadata = project.images[0].metadata

        self.assertEqual(metadata["added_at"], "")
        self.assertEqual(metadata["diagnosis"], "Не указано")
        self.assertEqual(metadata["notes"], "")

    def test_old_flat_project_records_are_not_migrated(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "name": "old",
                "images": [
                    {
                        "id": "record-id",
                        "path": "images/leaf.png",
                        "display_name": "leaf.png",
                    }
                ],
            }
        )

        self.assertEqual(project.images, [])


if __name__ == "__main__":
    unittest.main()
