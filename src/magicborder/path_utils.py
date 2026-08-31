from __future__ import annotations

import os
from pathlib import Path


def portable_path_reference(path: Path, base_dir: Path) -> str:
    try:
        reference_path = Path(os.path.relpath(path, base_dir))
    except ValueError:
        reference_path = path
    return reference_path.as_posix()


def annotation_image_candidates(
    image_reference: str, annotation_path: Path
) -> list[Path]:
    normalized_reference = image_reference.strip()
    if not normalized_reference:
        return []

    candidates: list[Path] = []
    for stored_path in _path_variants(normalized_reference):
        if stored_path.is_absolute():
            _append_unique(candidates, stored_path)
            continue

        annotation_dir = annotation_path.parent
        _append_unique(candidates, (annotation_dir / stored_path).resolve())
        if stored_path.name:
            _append_unique(candidates, (annotation_dir / stored_path.name).resolve())

    return candidates


def _path_variants(path_text: str) -> list[Path]:
    normalized = path_text.strip()
    variants = [Path(normalized)]
    if "\\" in normalized:
        variants.append(Path(normalized.replace("\\", "/")))

    unique_variants: list[Path] = []
    seen: set[str] = set()
    for variant in variants:
        variant_key = str(variant)
        if variant_key not in seen:
            unique_variants.append(variant)
            seen.add(variant_key)
    return unique_variants


def _append_unique(paths: list[Path], path: Path) -> None:
    if path not in paths:
        paths.append(path)
