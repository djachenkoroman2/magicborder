from __future__ import annotations

import argparse
import platform
import shlex
import shutil
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "build_tools" / "manifests" / "magicborder.toml"
SUPPORTED_PLATFORMS = {
    "linux-x86_64",
    "windows-x86_64",
    "macos-x86_64",
    "macos-arm64",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    manifest = _load_manifest(manifest_path)

    current_platform = _current_platform_id()
    requested_platform = current_platform if args.platform == "auto" else args.platform
    if requested_platform not in SUPPORTED_PLATFORMS:
        raise SystemExit(f"Неподдерживаемая целевая платформа: {requested_platform}")

    backend_name = args.backend or str(manifest["build"]["backend"])
    if backend_name != "pyinstaller":
        raise SystemExit(f"Неподдерживаемый backend: {backend_name}")

    output_root = Path(args.output_dir or manifest["build"]["output_root"])
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    if requested_platform != current_platform:
        message = (
            f"Текущая платформа: {current_platform}; запрошена: {requested_platform}. "
            "Portable-сборка должна выполняться на целевой ОС."
        )
        if not args.dry_run:
            raise SystemExit(message)
        print(f"WARNING: {message}")

    from backends import pyinstaller

    plan = pyinstaller.create_plan(
        repo_root=REPO_ROOT,
        manifest=manifest,
        platform_id=requested_platform,
        clean=args.clean,
    )
    _print_plan(
        manifest=manifest,
        platform_id=requested_platform,
        output_root=output_root,
        backend_name=backend_name,
        command=plan.command,
    )
    if args.dry_run:
        return 0

    bundle_path = pyinstaller.build(plan, clean=args.clean)
    archive_path = _package_bundle(
        bundle_path=bundle_path,
        manifest=manifest,
        platform_id=requested_platform,
        output_root=output_root,
        build_root=plan.build_root,
    )
    if not args.skip_smoke_test:
        _smoke_test_archive(archive_path, manifest=manifest, platform_id=requested_platform)
    print(f"Portable archive: {archive_path}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a portable MagicBorder application archive for the current platform.",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--backend", choices=["pyinstaller"], default=None)
    parser.add_argument(
        "--platform",
        choices=["auto", *sorted(SUPPORTED_PLATFORMS)],
        default="auto",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    for section in ("app", "python", "build"):
        if section not in payload or not isinstance(payload[section], dict):
            raise ValueError(f"Manifest должен содержать секцию [{section}].")
    for key in ("name", "version", "entrypoint"):
        if not str(payload["app"].get(key, "")).strip():
            raise ValueError(f"Manifest [app] должен содержать {key}.")
    if not isinstance(payload.get("data", []), list):
        raise ValueError("Manifest data должен быть списком [[data]].")
    for item in payload.get("data", []):
        source = REPO_ROOT / str(item.get("source", ""))
        if not source.exists():
            raise FileNotFoundError(f"Файл данных не найден: {source}")
        if not str(item.get("target", "")).strip():
            raise ValueError("Каждый [[data]] должен содержать target.")
    return payload


def _current_platform_id() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_x86_64 = machine in {"x86_64", "amd64"}
    if system == "linux" and is_x86_64:
        return "linux-x86_64"
    if system == "windows" and is_x86_64:
        return "windows-x86_64"
    if system == "darwin" and is_x86_64:
        return "macos-x86_64"
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "macos-arm64"
    raise SystemExit(f"Текущая платформа не поддержана: {system}/{machine}")


def _print_plan(
    *,
    manifest: dict[str, Any],
    platform_id: str,
    output_root: Path,
    backend_name: str,
    command: list[str],
) -> None:
    app = manifest["app"]
    archive_name = _archive_name(app["name"], app["version"], platform_id)
    print(f"App: {app.get('display_name', app['name'])} {app['version']}")
    print(f"Platform: {platform_id}")
    print(f"Backend: {backend_name}")
    print(f"Output: {output_root / platform_id / archive_name}")
    print(f"Command: {shlex.join(command)}")


def _package_bundle(
    *,
    bundle_path: Path,
    manifest: dict[str, Any],
    platform_id: str,
    output_root: Path,
    build_root: Path,
) -> Path:
    app = manifest["app"]
    app_name = str(app["name"])
    version = str(app["version"])
    package_name = f"{app_name}-{version}-{platform_id}"
    package_parent = build_root / "package"
    package_root = package_parent / package_name
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)

    destination = package_root / bundle_path.name
    if bundle_path.is_dir():
        shutil.copytree(bundle_path, destination)
    else:
        shutil.copy2(bundle_path, destination)

    platform_output = output_root / platform_id
    platform_output.mkdir(parents=True, exist_ok=True)
    archive_base = platform_output / package_name
    archive_format = "zip" if platform_id.startswith(("windows-", "macos-")) else "gztar"
    created = shutil.make_archive(
        str(archive_base),
        archive_format,
        root_dir=package_parent,
        base_dir=package_name,
    )
    return Path(created)


def _archive_name(app_name: str, version: str, platform_id: str) -> str:
    suffix = ".zip" if platform_id.startswith(("windows-", "macos-")) else ".tar.gz"
    return f"{app_name}-{version}-{platform_id}{suffix}"


def _smoke_test_archive(
    archive_path: Path,
    *,
    manifest: dict[str, Any],
    platform_id: str,
) -> None:
    if not archive_path.exists() or archive_path.stat().st_size <= 0:
        raise RuntimeError(f"Архив не создан или пуст: {archive_path}")

    members = _archive_members(archive_path)
    app_name = str(manifest["app"]["name"])
    executable_name = f"{app_name}.exe" if platform_id.startswith("windows-") else app_name
    if not any(PurePosixPath(name).name == executable_name for name in members):
        raise RuntimeError(f"В архиве не найден исполняемый файл {executable_name}.")
    if not any(PurePosixPath(name).name == "oiv_ampelometric_scales.json" for name in members):
        raise RuntimeError("В архиве не найден oiv_ampelometric_scales.json.")
    if not any("magicborder/assets/" in name or "magicborder/assets\\" in name for name in members):
        raise RuntimeError("В архиве не найдены assets MagicBorder.")


def _archive_members(archive_path: Path) -> list[str]:
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            return archive.namelist()
    if archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            return archive.getnames()
    raise RuntimeError(f"Неподдерживаемый формат архива: {archive_path}")


if __name__ == "__main__":
    raise SystemExit(main())

