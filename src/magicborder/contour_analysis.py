from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .histograms import (
    HistogramPlotData,
    build_hsv_histogram,
    build_lab_histogram,
    build_lms_histogram,
    build_rgb_histogram,
    build_yuv_histogram,
    rgb_to_lms,
)
from .models import Point

ContourSignature = tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class ContourColorStats:
    mean_rgb: tuple[int, int, int]
    mean_lab: tuple[int, int, int]
    mean_hsv: tuple[int, int, int]
    mean_yuv: tuple[int, int, int]
    mean_lms: tuple[int, int, int]
    pixel_count: int


@dataclass(frozen=True, slots=True)
class ContourHistogramData:
    rgb: HistogramPlotData | None
    lab: HistogramPlotData | None
    hsv: HistogramPlotData | None
    yuv: HistogramPlotData | None
    lms: HistogramPlotData | None


@dataclass(frozen=True, slots=True)
class ContourAnalysis:
    stats: ContourColorStats
    histograms: ContourHistogramData


def contour_signature(points: list[Point]) -> ContourSignature:
    return tuple(
        (round(float(point.x), 3), round(float(point.y), 3)) for point in points
    )


def contour_rgb_pixels_from_points(
    rgb_array: np.ndarray, points: list[Point]
) -> np.ndarray:
    if len(points) < 3:
        return np.empty((0, 3), dtype=np.uint8)

    mask = np.zeros(rgb_array.shape[:2], dtype=np.uint8)
    polygon = np.array(
        [[int(round(point.x)), int(round(point.y))] for point in points],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [polygon], 255)
    pixels = rgb_array[mask > 0]
    return np.ascontiguousarray(pixels.reshape((-1, 3)))


def build_contour_analysis(
    rgb_array: np.ndarray, points: list[Point]
) -> ContourAnalysis | None:
    pixels = contour_rgb_pixels_from_points(rgb_array, points)
    if pixels.size == 0:
        return None

    return ContourAnalysis(
        stats=build_contour_color_stats(pixels),
        histograms=ContourHistogramData(
            rgb=build_rgb_histogram(pixels),
            lab=build_lab_histogram(pixels),
            hsv=build_hsv_histogram(pixels),
            yuv=build_yuv_histogram(pixels),
            lms=build_lms_histogram(pixels),
        ),
    )


def build_contour_color_stats(rgb_pixels: np.ndarray) -> ContourColorStats:
    mean_values = np.rint(rgb_pixels.mean(axis=0)).astype(int)
    lms_values = rgb_to_lms(rgb_pixels)
    return ContourColorStats(
        mean_rgb=(int(mean_values[0]), int(mean_values[1]), int(mean_values[2])),
        mean_lab=_mean_lab_values(rgb_pixels),
        mean_hsv=_mean_hsv_values(rgb_pixels),
        mean_yuv=_mean_yuv_values(rgb_pixels),
        mean_lms=_mean_lms_values_from_total(
            lms_values.sum(axis=0),
            lms_values.max(axis=0),
            int(lms_values.shape[0]),
        ),
        pixel_count=int(rgb_pixels.shape[0]),
    )


def lab_values_from_rgb_pixels(rgb_pixels: np.ndarray) -> np.ndarray:
    lab_pixels = cv2.cvtColor(
        rgb_pixels.reshape((-1, 1, 3)), cv2.COLOR_RGB2LAB
    ).reshape((-1, 3))
    lab_pixels = lab_pixels.astype(np.float32)
    return np.column_stack(
        (
            lab_pixels[:, 0] * (100.0 / 255.0),
            lab_pixels[:, 1] - 128.0,
            lab_pixels[:, 2] - 128.0,
        )
    )


def hsv_values_from_rgb_pixels(rgb_pixels: np.ndarray) -> np.ndarray:
    hsv_pixels = cv2.cvtColor(
        rgb_pixels.reshape((-1, 1, 3)), cv2.COLOR_RGB2HSV
    ).reshape((-1, 3))
    hsv_values = hsv_pixels.astype(np.float32)
    hsv_values[:, 0] *= 2.0
    return hsv_values


def yuv_values_from_rgb_pixels(rgb_pixels: np.ndarray) -> np.ndarray:
    yuv_pixels = cv2.cvtColor(
        rgb_pixels.reshape((-1, 1, 3)), cv2.COLOR_RGB2YUV
    ).reshape((-1, 3))
    return yuv_pixels.astype(np.float32)


def mean_lms_values_from_total(
    lms_total: np.ndarray,
    lms_max: np.ndarray,
    pixel_count: int,
) -> tuple[int, int, int]:
    return _mean_lms_values_from_total(lms_total, lms_max, pixel_count)


def _mean_lab_values(rgb_pixels: np.ndarray) -> tuple[int, int, int]:
    lab_values = lab_values_from_rgb_pixels(rgb_pixels)
    mean_values = np.rint(lab_values.mean(axis=0)).astype(int)
    return int(mean_values[0]), int(mean_values[1]), int(mean_values[2])


def _mean_hsv_values(rgb_pixels: np.ndarray) -> tuple[int, int, int]:
    hsv_values = hsv_values_from_rgb_pixels(rgb_pixels)
    mean_values = np.rint(hsv_values.mean(axis=0)).astype(int)
    return int(mean_values[0]), int(mean_values[1]), int(mean_values[2])


def _mean_yuv_values(rgb_pixels: np.ndarray) -> tuple[int, int, int]:
    yuv_values = yuv_values_from_rgb_pixels(rgb_pixels)
    mean_values = np.rint(yuv_values.mean(axis=0)).astype(int)
    return int(mean_values[0]), int(mean_values[1]), int(mean_values[2])


def _mean_lms_values_from_total(
    lms_total: np.ndarray,
    lms_max: np.ndarray,
    pixel_count: int,
) -> tuple[int, int, int]:
    max_values = np.maximum(lms_max, 1e-9)
    normalized_mean = (lms_total / max(1, pixel_count)) / max_values * 255.0
    mean_values = np.rint(normalized_mean).astype(int)
    mean_values = np.clip(mean_values, 0, 255)
    return int(mean_values[0]), int(mean_values[1]), int(mean_values[2])
