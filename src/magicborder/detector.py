from __future__ import annotations

import cv2
import numpy as np

from .models import Point


def detect_leaf_contour(rgb_image: np.ndarray) -> list[Point]:
    """Detect the main leaf boundary and return it as editable contour points."""
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Ожидается цветное RGB-изображение.")

    working_image, scale = _resize_for_detection(rgb_image)
    image_bgr = cv2.cvtColor(working_image, cv2.COLOR_RGB2BGR)

    mask = _grabcut_mask(image_bgr)
    if not _mask_is_reasonable(mask):
        mask = _fallback_color_mask(image_bgr)
    mask = _postprocess_mask(mask)

    contour = _extract_largest_contour(mask)
    if contour is None:
        raise ValueError("Не удалось определить контур листа. Попробуйте изображение с более контрастным фоном.")

    points = _simplify_contour(contour)
    if scale != 1.0:
        points = points / scale

    return [Point(float(x), float(y)) for x, y in points]


def _resize_for_detection(rgb_image: np.ndarray, max_side: int = 1400) -> tuple[np.ndarray, float]:
    height, width = rgb_image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_side:
        return rgb_image, 1.0

    scale = max_side / float(largest_side)
    resized = cv2.resize(
        rgb_image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _grabcut_mask(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    margin_x = max(6, int(width * 0.05))
    margin_y = max(6, int(height * 0.05))
    rect_width = width - margin_x * 2
    rect_height = height - margin_y * 2
    if rect_width <= 4 or rect_height <= 4:
        return np.zeros((height, width), dtype=np.uint8)

    mask = np.zeros((height, width), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    rect = (margin_x, margin_y, rect_width, rect_height)

    try:
        cv2.grabCut(image_bgr, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((height, width), dtype=np.uint8)

    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    return foreground


def _fallback_color_mask(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    red, green, blue = cv2.split(image_rgb)
    excess_green = 2.0 * green - red - blue
    excess_green = cv2.normalize(excess_green, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    _, exg_mask = cv2.threshold(excess_green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, saturation_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    combined = cv2.bitwise_and(exg_mask, saturation_mask)
    if not _mask_is_reasonable(combined):
        combined = cv2.bitwise_or(exg_mask, saturation_mask)
    return combined


def _mask_is_reasonable(mask: np.ndarray) -> bool:
    foreground_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    return 0.01 <= foreground_ratio <= 0.95


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    kernel_size = max(3, int(round(min(mask.shape[:2]) * 0.01)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contour = _extract_largest_contour(cleaned)
    if contour is None:
        return cleaned

    filled = np.zeros_like(cleaned)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    return filled


def _extract_largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    min_area = max(64.0, mask.shape[0] * mask.shape[1] * 0.001)
    if cv2.contourArea(contour) < min_area:
        return None
    return contour


def _simplify_contour(contour: np.ndarray, min_points: int = 16, max_points: int = 80) -> np.ndarray:
    perimeter = cv2.arcLength(contour, True)
    source_points = contour[:, 0, :].astype(np.float32)
    candidate = source_points

    for ratio in np.linspace(0.001, 0.02, 30):
        approx = cv2.approxPolyDP(contour, epsilon=float(perimeter * ratio), closed=True)
        approx_points = approx[:, 0, :].astype(np.float32)
        candidate = approx_points
        if min_points <= len(approx_points) <= max_points:
            return approx_points

    if len(candidate) > max_points:
        return _resample_closed_polyline(candidate, max_points)
    if len(candidate) < min_points:
        return _resample_closed_polyline(source_points, min_points)
    return candidate


def _resample_closed_polyline(points: np.ndarray, target_count: int) -> np.ndarray:
    points = points.astype(np.float32)
    if len(points) == 0:
        return points
    if len(points) == 1:
        return np.repeat(points, target_count, axis=0)

    closed_points = np.vstack([points, points[0]])
    segments = np.diff(closed_points, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 1e-6:
        return points[:target_count]

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    targets = np.linspace(0.0, total_length, target_count, endpoint=False)
    resampled: list[np.ndarray] = []

    for target in targets:
        segment_index = int(np.searchsorted(cumulative, target, side="right") - 1)
        segment_index = max(0, min(segment_index, len(points) - 1))
        next_index = (segment_index + 1) % len(points)
        segment_length = float(segment_lengths[segment_index])

        if segment_length <= 1e-6:
            resampled.append(points[segment_index])
            continue

        local_distance = target - cumulative[segment_index]
        local_ratio = float(local_distance / segment_length)
        interpolated = points[segment_index] + local_ratio * (points[next_index] - points[segment_index])
        resampled.append(interpolated)

    return np.vstack(resampled).astype(np.float32)
