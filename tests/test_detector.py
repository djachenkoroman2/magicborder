from __future__ import annotations

import cv2
import numpy as np
import pytest

from magicborder import detector
from magicborder.detector import (
    _extract_largest_contour,
    _fallback_color_mask,
    _grabcut_mask,
    _mask_is_reasonable,
    _resample_closed_polyline,
    _resize_for_detection,
    _simplify_contour,
    detect_leaf_contour,
)


def _blob_image(width: int = 200, height: int = 160) -> np.ndarray:
    """Тёмно-синий фон с крупным зелёным пятном по центру."""
    image = np.full((height, width, 3), (20, 20, 90), dtype=np.uint8)
    cv2.ellipse(
        image,
        (width // 2, height // 2),
        (width // 4, height // 4),
        0,
        0,
        360,
        (40, 200, 60),
        thickness=-1,
    )
    return image


def _circle_contour(radius: float = 100.0, point_count: int = 240) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    points = np.stack(
        [
            np.round(radius + radius * np.cos(angles)),
            np.round(radius + radius * np.sin(angles)),
        ],
        axis=1,
    ).astype(np.int32)
    return points.reshape((-1, 1, 2))


def _square_contour(side: int = 100) -> np.ndarray:
    top = [(x, 0) for x in range(side)]
    right = [(side - 1, y) for y in range(side)]
    bottom = [(x, side - 1) for x in range(side - 1, -1, -1)]
    left = [(0, y) for y in range(side - 1, -1, -1)]
    return np.array(top + right + bottom + left, dtype=np.int32).reshape((-1, 1, 2))


class TestDetectLeafContourInput:
    def test_grayscale_image_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="RGB"):
            detect_leaf_contour(np.zeros((32, 32), dtype=np.uint8))

    def test_rgba_image_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="RGB"):
            detect_leaf_contour(np.zeros((32, 32, 4), dtype=np.uint8))

    def test_contrasting_blob_produces_contour_points(self) -> None:
        points = detect_leaf_contour(_blob_image())

        assert len(points) >= 3
        assert all(0 <= point.x <= 200 and 0 <= point.y <= 160 for point in points)

    def test_missing_contour_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        empty_mask = np.zeros((60, 60), dtype=np.uint8)
        monkeypatch.setattr(detector, "_grabcut_mask", lambda _image: empty_mask)
        monkeypatch.setattr(detector, "_fallback_color_mask", lambda _image: empty_mask)

        with pytest.raises(ValueError, match="Не удалось определить контур листа"):
            detect_leaf_contour(np.zeros((60, 60, 3), dtype=np.uint8))

    def test_unreasonable_grabcut_mask_falls_back_to_color_mask(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []
        original_fallback = detector._fallback_color_mask

        def spy_fallback(image_bgr: np.ndarray) -> np.ndarray:
            calls.append("fallback")
            return original_fallback(image_bgr)

        monkeypatch.setattr(
            detector,
            "_grabcut_mask",
            lambda image: np.zeros(image.shape[:2], dtype=np.uint8),
        )
        monkeypatch.setattr(detector, "_fallback_color_mask", spy_fallback)

        detect_leaf_contour(_blob_image())

        assert calls == ["fallback"]

    def test_large_image_points_are_scaled_back_to_source_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        image = np.full((900, 2000, 3), (20, 20, 90), dtype=np.uint8)
        cv2.rectangle(image, (1200, 200), (1900, 700), (40, 200, 60), thickness=-1)

        def mask_from_green(image_bgr: np.ndarray) -> np.ndarray:
            green = image_bgr[:, :, 1]
            return np.where(green > 120, 255, 0).astype(np.uint8)

        monkeypatch.setattr(detector, "_grabcut_mask", mask_from_green)

        points = detect_leaf_contour(image)

        # Без обратного деления на scale координаты не превысили бы 1400 px.
        assert max(point.x for point in points) > 1400
        assert max(point.x for point in points) <= 2000


class TestResizeForDetection:
    def test_small_image_is_returned_unchanged(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        resized, scale = _resize_for_detection(image)

        assert scale == 1.0
        assert resized is image

    def test_boundary_size_is_not_resized(self) -> None:
        image = np.zeros((10, 1400, 3), dtype=np.uint8)

        resized, scale = _resize_for_detection(image)

        assert scale == 1.0
        assert resized.shape == image.shape

    def test_large_image_is_downscaled(self) -> None:
        image = np.zeros((1000, 2800, 3), dtype=np.uint8)

        resized, scale = _resize_for_detection(image)

        assert scale == pytest.approx(0.5)
        assert resized.shape[:2] == (500, 1400)


class TestMaskIsReasonable:
    @pytest.mark.parametrize(
        ("foreground_pixels", "expected"),
        [
            (99, False),
            (100, True),
            (9500, True),
            (9501, False),
        ],
    )
    def test_foreground_ratio_boundaries(self, foreground_pixels: int, expected: bool) -> None:
        mask = np.zeros(10_000, dtype=np.uint8)
        mask[:foreground_pixels] = 255

        assert _mask_is_reasonable(mask.reshape((100, 100))) is expected


class TestGrabCutMask:
    def test_tiny_image_returns_empty_mask(self) -> None:
        image_bgr = np.zeros((10, 10, 3), dtype=np.uint8)

        mask = _grabcut_mask(image_bgr)

        assert mask.shape == (10, 10)
        assert not mask.any()

    def test_narrow_image_returns_empty_mask(self) -> None:
        image_bgr = np.zeros((60, 13, 3), dtype=np.uint8)

        mask = _grabcut_mask(image_bgr)

        assert not mask.any()

    def test_opencv_error_returns_empty_mask(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_cv_error(*_args, **_kwargs):
            raise cv2.error("grabCut failed")

        monkeypatch.setattr(cv2, "grabCut", raise_cv_error)

        mask = _grabcut_mask(np.zeros((60, 60, 3), dtype=np.uint8))

        assert mask.shape == (60, 60)
        assert not mask.any()


class TestFallbackColorMask:
    def test_unreasonable_intersection_switches_to_union(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        union_results: list[np.ndarray] = []
        original_bitwise_or = cv2.bitwise_or

        def spy_bitwise_or(first, second, *args, **kwargs):
            result = original_bitwise_or(first, second, *args, **kwargs)
            union_results.append(result)
            return result

        monkeypatch.setattr(detector, "_mask_is_reasonable", lambda _mask: False)
        monkeypatch.setattr(cv2, "bitwise_or", spy_bitwise_or)

        image_bgr = cv2.cvtColor(_blob_image(), cv2.COLOR_RGB2BGR)
        mask = _fallback_color_mask(image_bgr)

        assert len(union_results) == 1
        assert np.array_equal(mask, union_results[0])


class TestExtractLargestContour:
    def test_empty_mask_returns_none(self) -> None:
        assert _extract_largest_contour(np.zeros((50, 50), dtype=np.uint8)) is None

    def test_contour_below_min_area_returns_none(self) -> None:
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[0:5, 0:5] = 255  # площадь заметно меньше max(64, 200*200*0.001) = 64

        assert _extract_largest_contour(mask) is None

    def test_largest_contour_is_selected(self) -> None:
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        mask[60:160, 60:160] = 255

        contour = _extract_largest_contour(mask)

        assert contour is not None
        assert cv2.contourArea(contour) > 9_000


class TestSimplifyContour:
    def test_circle_is_simplified_within_bounds(self) -> None:
        points = _simplify_contour(_circle_contour())

        assert 16 <= len(points) <= 80

    def test_too_few_points_are_resampled_to_min_points(self) -> None:
        points = _simplify_contour(_square_contour())

        assert len(points) == 16

    def test_too_many_points_are_resampled_to_max_points(self) -> None:
        # У квадрата аппроксимация всегда даёт 4 точки, поэтому цикл по ratio
        # не находит попадания в [2, 3] и срабатывает ветка max_points.
        points = _simplify_contour(_square_contour(), min_points=2, max_points=3)

        assert len(points) == 3


class TestResampleClosedPolyline:
    def test_empty_input_is_returned_as_is(self) -> None:
        points = _resample_closed_polyline(np.empty((0, 2), dtype=np.float32), 10)

        assert points.shape == (0, 2)

    def test_single_point_is_repeated(self) -> None:
        points = _resample_closed_polyline(np.array([[3.0, 4.0]], dtype=np.float32), 5)

        assert points.shape == (5, 2)
        assert np.array_equal(points, np.repeat([[3.0, 4.0]], 5, axis=0))

    def test_zero_total_length_is_truncated(self) -> None:
        source = np.zeros((6, 2), dtype=np.float32)

        points = _resample_closed_polyline(source, 4)

        assert points.shape == (4, 2)
        assert not points.any()

    def test_duplicated_points_do_not_break_resampling(self) -> None:
        source = np.array(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
            dtype=np.float32,
        )

        points = _resample_closed_polyline(source, 8)

        assert points.shape == (8, 2)
        assert np.isfinite(points).all()

    def test_square_is_resampled_evenly(self) -> None:
        source = np.array(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            dtype=np.float32,
        )

        points = _resample_closed_polyline(source, 8)

        assert points.shape == (8, 2)
        assert np.allclose(points[0], [0.0, 0.0])
        assert np.allclose(points[2], [10.0, 0.0])
