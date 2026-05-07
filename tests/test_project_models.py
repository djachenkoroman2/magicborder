from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from magicborder.io_utils import load_project, save_project
from magicborder.models import Annotation, Point, ProjectDocument, ProjectImageRecord


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

            loaded_project = load_project(project_path)

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
                "name": "broken",
                "images": [
                    {
                        "id": "leaf-1",
                        "path": "images/leaf.png",
                        "display_name": "leaf.png",
                        "annotation": {"points": []},
                    }
                ],
            }
        )

        record = project.images[0]

        self.assertIsNone(record.annotation)
        self.assertIsNotNone(record.annotation_error)
        self.assertEqual(record.to_dict()["annotation"], {"points": []})

    def test_project_image_metadata_round_trip_preserves_unknown_fields(self) -> None:
        project = ProjectDocument(
            name="metadata",
            images=[
                ProjectImageRecord(
                    id="leaf-1",
                    relative_path="images/leaf.png",
                    display_name="leaf.png",
                    metadata={
                        "added_at": "2026-05-03T10:20:30+03:00",
                        "humidity": "55",
                        "diagnosis": "leaf_blight",
                        "custom_field": "kept",
                    },
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "metadata.json"
            save_project(project_path, project)

            loaded_project = load_project(project_path)

        metadata = loaded_project.images[0].metadata
        self.assertEqual(metadata["added_at"], "2026-05-03T10:20:30+03:00")
        self.assertEqual(metadata["humidity"], "55")
        self.assertEqual(metadata["diagnosis"], "leaf_blight")
        self.assertEqual(metadata["custom_field"], "kept")
        self.assertIn("captured_at", metadata)

    def test_project_image_metadata_defaults_for_old_project(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "name": "old",
                "images": [
                    {
                        "id": "leaf-1",
                        "path": "images/leaf.png",
                        "display_name": "leaf.png",
                    }
                ],
            }
        )

        metadata = project.images[0].metadata

        self.assertEqual(metadata["added_at"], "")
        self.assertNotIn("sample_id", metadata)
        self.assertEqual(metadata["diagnosis"], "Не указано")
        self.assertEqual(metadata["notes"], "")

    def test_legacy_sample_id_is_preserved_but_not_required(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "name": "legacy",
                "images": [
                    {
                        "id": "record-id",
                        "path": "images/leaf.png",
                        "display_name": "leaf.png",
                        "metadata": {"sample_id": "legacy-sample"},
                    }
                ],
            }
        )

        record = project.images[0]

        self.assertEqual(record.id, "record-id")
        self.assertEqual(record.metadata["sample_id"], "legacy-sample")

    def test_legacy_sample_id_can_seed_missing_record_id(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "name": "legacy",
                "images": [
                    {
                        "path": "images/leaf.png",
                        "display_name": "leaf.png",
                        "metadata": {"sample_id": "legacy-sample"},
                    }
                ],
            }
        )

        self.assertEqual(project.images[0].id, "legacy-sample")


if __name__ == "__main__":
    unittest.main()
