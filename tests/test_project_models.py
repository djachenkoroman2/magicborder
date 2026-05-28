from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from magicborder.io_utils import load_project, save_project
from magicborder.models import (
    PROJECT_FORMAT_VERSION,
    Annotation,
    ImageCalibration,
    Point,
    ProjectAngleMeasurement,
    ProjectDocument,
    ProjectImageRecord,
    ProjectImageMeasurements,
    ProjectInfo,
)


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

    def test_project_round_trip_preserves_image_calibration(self) -> None:
        calibration = ImageCalibration(
            start=Point(1, 2),
            end=Point(121, 2),
            length_mm=10,
        )
        project = ProjectDocument(
            name="calibration",
            images=[
                ProjectImageRecord(
                    id="leaf-1",
                    relative_path="images/leaf.png",
                    display_name="leaf.png",
                    calibration=calibration,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "calibration.json"
            save_project(project_path, project)
            payload = json.loads(project_path.read_text(encoding="utf-8"))

            loaded_project = load_project(project_path)

        record_payload = payload["images"][0]
        self.assertEqual(
            record_payload["calibration"],
            {
                "start": {"x": 1.0, "y": 2.0},
                "end": {"x": 121.0, "y": 2.0},
                "length_mm": 10.0,
            },
        )
        loaded_calibration = loaded_project.images[0].calibration
        self.assertIsNotNone(loaded_calibration)
        self.assertEqual(loaded_calibration.start, Point(1, 2))
        self.assertEqual(loaded_calibration.end, Point(121, 2))
        self.assertEqual(loaded_calibration.length_mm, 10.0)
        self.assertEqual(loaded_calibration.pixel_length(), 120.0)
        self.assertEqual(loaded_calibration.pixels_per_mm(), 12.0)

    def test_project_round_trip_preserves_image_angle_measurements(self) -> None:
        project = ProjectDocument(
            name="measurements",
            images=[
                ProjectImageRecord(
                    id="leaf-1",
                    relative_path="images/leaf.png",
                    display_name="leaf.png",
                    measurements=ProjectImageMeasurements(
                        angles=[
                            ProjectAngleMeasurement(
                                id="angle-1",
                                first=Point(2, 10),
                                vertex=Point(2, 2),
                                second=Point(10, 2),
                                note="контрольный угол",
                            )
                        ]
                    ),
                )
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "measurements.json"
            save_project(project_path, project)
            payload = json.loads(project_path.read_text(encoding="utf-8"))

            loaded_project = load_project(project_path)

        record_payload = payload["images"][0]
        self.assertEqual(
            record_payload["measurements"]["angles"],
            [
                {
                    "id": "angle-1",
                    "first": {"x": 2.0, "y": 10.0},
                    "vertex": {"x": 2.0, "y": 2.0},
                    "second": {"x": 10.0, "y": 2.0},
                    "note": "контрольный угол",
                }
            ],
        )
        loaded_angle = loaded_project.images[0].measurements.angles[0]
        self.assertEqual(loaded_angle.id, "angle-1")
        self.assertEqual(loaded_angle.first, Point(2, 10))
        self.assertEqual(loaded_angle.vertex, Point(2, 2))
        self.assertEqual(loaded_angle.second, Point(10, 2))
        self.assertEqual(loaded_angle.note, "контрольный угол")

    def test_project_keeps_corrupt_calibration_payload_without_breaking_load(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "version": PROJECT_FORMAT_VERSION,
                "name": "broken_calibration",
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
                        "calibration": {
                            "start": {"x": 1, "y": 1},
                            "end": {"x": 1, "y": 1},
                            "length_mm": 10,
                        },
                    }
                ],
            }
        )

        record = project.images[0]

        self.assertIsNone(record.calibration)
        self.assertIsNotNone(record.calibration_error)
        self.assertEqual(
            record.to_dict()["calibration"],
            {
                "start": {"x": 1, "y": 1},
                "end": {"x": 1, "y": 1},
                "length_mm": 10,
            },
        )

    def test_project_loads_images_without_measurements_as_empty_collection(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "version": PROJECT_FORMAT_VERSION,
                "name": "old_project",
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

        record = project.images[0]

        self.assertEqual(record.measurements.angles, [])
        self.assertNotIn("measurements", record.to_dict())

    def test_project_info_round_trip_uses_grouped_json(self) -> None:
        project = ProjectDocument(
            name="leaves",
            images=[],
            project_info=ProjectInfo(general_info="Полевой эксперимент"),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "leaves.json"
            save_project(project_path, project)
            payload = json.loads(project_path.read_text(encoding="utf-8"))

            loaded_project = load_project(project_path)

        self.assertEqual(payload["project_info"], {"general_info": "Полевой эксперимент"})
        self.assertEqual(loaded_project.project_info.general_info, "Полевой эксперимент")

    def test_project_info_defaults_for_old_json(self) -> None:
        project = ProjectDocument.from_dict(
            {
                "version": PROJECT_FORMAT_VERSION,
                "name": "old",
                "images": [],
            }
        )

        self.assertEqual(project.project_info.general_info, "")

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
