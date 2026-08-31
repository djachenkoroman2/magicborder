from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import stat
import subprocess
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
SUPPORTED_ARTIFACTS = {"archive", "appimage", "onefile"}
ARTIFACT_PLATFORMS = {
    "archive": SUPPORTED_PLATFORMS,
    "appimage": {"linux-x86_64"},
    "onefile": {"windows-x86_64"},
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
    _validate_artifact_platform(args.artifact, requested_platform)

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

    appimagetool_path = None
    if args.artifact == "appimage" and not args.dry_run:
        appimagetool_path = _resolve_appimagetool(args.appimagetool)

    pyinstaller = _load_pyinstaller_backend()
    bundle_mode = "onefile" if args.artifact == "onefile" else "onedir"
    plan = pyinstaller.create_plan(
        repo_root=REPO_ROOT,
        manifest=manifest,
        platform_id=requested_platform,
        clean=args.clean,
        bundle_mode=bundle_mode,
    )
    artifact_path = _artifact_output_path(
        manifest=manifest,
        platform_id=requested_platform,
        output_root=output_root,
        artifact=args.artifact,
    )
    _print_plan(
        manifest=manifest,
        platform_id=requested_platform,
        output_path=artifact_path,
        backend_name=backend_name,
        artifact=args.artifact,
        command=plan.command,
    )
    if args.dry_run:
        return 0

    bundle_path = pyinstaller.build(plan, clean=args.clean)
    if args.artifact == "archive":
        artifact_path = _package_archive(
            bundle_path=bundle_path,
            manifest=manifest,
            platform_id=requested_platform,
            output_root=output_root,
            build_root=plan.build_root,
        )
    elif args.artifact == "onefile":
        artifact_path = _package_onefile(
            bundle_path=bundle_path,
            manifest=manifest,
            platform_id=requested_platform,
            output_root=output_root,
        )
    else:
        artifact_path = _package_appimage(
            bundle_path=bundle_path,
            manifest=manifest,
            platform_id=requested_platform,
            output_root=output_root,
            build_root=plan.build_root,
            appimagetool=appimagetool_path,
        )

    if not args.skip_smoke_test:
        _smoke_test_artifact(
            artifact_path,
            manifest=manifest,
            platform_id=requested_platform,
            artifact=args.artifact,
        )
    print(f"Portable {args.artifact}: {artifact_path}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a portable MagicBorder application artifact for the current platform.",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--backend", choices=["pyinstaller"], default=None)
    parser.add_argument(
        "--platform",
        choices=["auto", *sorted(SUPPORTED_PLATFORMS)],
        default="auto",
    )
    parser.add_argument(
        "--artifact",
        choices=sorted(SUPPORTED_ARTIFACTS),
        default="archive",
    )
    parser.add_argument("--appimagetool", default=None)
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
        source = _repo_path(str(item.get("source", "")))
        if not source.exists():
            raise FileNotFoundError(f"Файл данных не найден: {source}")
        if not str(item.get("target", "")).strip():
            raise ValueError("Каждый [[data]] должен содержать target.")

    appimage = payload.get("appimage")
    if appimage is not None and not isinstance(appimage, dict):
        raise ValueError("Manifest [appimage] должен быть секцией TOML.")
    if isinstance(appimage, dict):
        icon = str(appimage.get("icon", "")).strip()
        if icon and not _repo_path(icon).exists():
            raise FileNotFoundError(f"Иконка AppImage не найдена: {_repo_path(icon)}")
        categories = appimage.get("desktop_categories", [])
        if categories and not isinstance(categories, list):
            raise ValueError(
                "Manifest [appimage].desktop_categories должен быть списком."
            )
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


def _validate_artifact_platform(artifact: str, platform_id: str) -> None:
    if artifact not in SUPPORTED_ARTIFACTS:
        raise SystemExit(f"Неподдерживаемый тип артефакта: {artifact}")
    allowed_platforms = ARTIFACT_PLATFORMS[artifact]
    if platform_id not in allowed_platforms:
        supported = ", ".join(sorted(allowed_platforms))
        raise SystemExit(
            f"Artifact {artifact} не поддерживается для платформы {platform_id}. "
            f"Поддерживаемые платформы: {supported}."
        )


def _load_pyinstaller_backend() -> Any:
    try:
        from build_tools.backends import pyinstaller
    except ModuleNotFoundError:
        from backends import pyinstaller
    return pyinstaller


def _print_plan(
    *,
    manifest: dict[str, Any],
    platform_id: str,
    output_path: Path,
    backend_name: str,
    artifact: str,
    command: list[str],
) -> None:
    app = manifest["app"]
    print(f"App: {app.get('display_name', app['name'])} {app['version']}")
    print(f"Platform: {platform_id}")
    print(f"Artifact: {artifact}")
    print(f"Backend: {backend_name}")
    print(f"Output: {output_path}")
    print(f"Command: {shlex.join(command)}")


def _package_archive(
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
    archive_format = (
        "zip" if platform_id.startswith(("windows-", "macos-")) else "gztar"
    )
    created = shutil.make_archive(
        str(archive_base),
        archive_format,
        root_dir=package_parent,
        base_dir=package_name,
    )
    return Path(created)


def _package_onefile(
    *,
    bundle_path: Path,
    manifest: dict[str, Any],
    platform_id: str,
    output_root: Path,
) -> Path:
    if not bundle_path.is_file():
        raise RuntimeError(
            f"PyInstaller onefile должен вернуть файл, получено: {bundle_path}"
        )

    output_path = _artifact_output_path(
        manifest=manifest,
        platform_id=platform_id,
        output_root=output_root,
        artifact="onefile",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle_path, output_path)
    return output_path


def _package_appimage(
    *,
    bundle_path: Path,
    manifest: dict[str, Any],
    platform_id: str,
    output_root: Path,
    build_root: Path,
    appimagetool: Path | str | None,
) -> Path:
    if platform_id != "linux-x86_64":
        raise RuntimeError(
            f"AppImage поддержан только для linux-x86_64, получено: {platform_id}"
        )

    tool_path = (
        appimagetool
        if isinstance(appimagetool, Path)
        else _resolve_appimagetool(appimagetool)
    )
    app_dir = _prepare_app_dir(
        bundle_path=bundle_path, manifest=manifest, build_root=build_root
    )
    output_path = _artifact_output_path(
        manifest=manifest,
        platform_id=platform_id,
        output_root=output_root,
        artifact="appimage",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    env = os.environ.copy()
    env.setdefault("ARCH", "x86_64")
    subprocess.run(
        [str(tool_path), str(app_dir), str(output_path)], check=True, env=env
    )
    if not output_path.exists():
        raise RuntimeError(f"appimagetool не создал ожидаемый файл: {output_path}")
    _add_executable_bits(output_path)
    return output_path


def _prepare_app_dir(
    *, bundle_path: Path, manifest: dict[str, Any], build_root: Path
) -> Path:
    app = manifest["app"]
    app_name = str(app["name"])
    app_dir = build_root / "appimage" / _app_dir_name(manifest)
    if app_dir.exists():
        shutil.rmtree(app_dir)

    bundle_destination = app_dir / "usr" / "bin" / app_name
    bundle_destination.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.is_dir():
        shutil.copytree(bundle_path, bundle_destination)
        executable_name = app_name
    else:
        bundle_destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle_path, bundle_destination / bundle_path.name)
        executable_name = bundle_path.name

    _write_apprun(
        app_dir / "AppRun", app_name=app_name, executable_name=executable_name
    )
    _write_desktop_entries(app_dir, manifest=manifest)
    _copy_appimage_icon(app_dir, manifest=manifest)
    return app_dir


def _write_apprun(path: Path, *, app_name: str, executable_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                'HERE="$(dirname "$(readlink -f "$0")")"',
                f'exec "$HERE/usr/bin/{app_name}/{executable_name}" "$@"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _add_executable_bits(path)


def _write_desktop_entries(app_dir: Path, *, manifest: dict[str, Any]) -> None:
    app = manifest["app"]
    app_name = str(app["name"])
    display_name = str(app.get("display_name", app_name))
    categories = _appimage_categories(manifest)
    desktop_text = "\n".join(
        [
            "[Desktop Entry]",
            "Type=Application",
            f"Name={display_name}",
            f"Exec={app_name}",
            f"Icon={app_name}",
            f"Categories={categories}",
            "Terminal=false",
            "",
        ]
    )

    root_desktop = app_dir / f"{app_name}.desktop"
    root_desktop.write_text(desktop_text, encoding="utf-8")

    share_desktop = app_dir / "usr" / "share" / "applications" / f"{app_name}.desktop"
    share_desktop.parent.mkdir(parents=True, exist_ok=True)
    share_desktop.write_text(desktop_text, encoding="utf-8")


def _copy_appimage_icon(app_dir: Path, *, manifest: dict[str, Any]) -> None:
    app_name = str(manifest["app"]["name"])
    icon_source = _appimage_icon_source(manifest)
    icon_suffix = icon_source.suffix or ".svg"

    root_icon = app_dir / f"{app_name}{icon_suffix}"
    root_icon.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(icon_source, root_icon)

    share_icon = (
        app_dir
        / "usr"
        / "share"
        / "icons"
        / "hicolor"
        / "scalable"
        / "apps"
        / f"{app_name}{icon_suffix}"
    )
    share_icon.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(icon_source, share_icon)


def _resolve_appimagetool(raw_path: str | None) -> Path:
    candidates = [raw_path, os.environ.get("APPIMAGETOOL")]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = _resolve_executable(candidate)
        if resolved is not None:
            return resolved
        raise RuntimeError(
            "appimagetool не найден. Укажите путь через --appimagetool, "
            "переменную APPIMAGETOOL или добавьте appimagetool в PATH."
        )

    found = shutil.which("appimagetool")
    if found:
        return Path(found)
    raise RuntimeError(
        "appimagetool не найден. Укажите путь через --appimagetool, "
        "переменную APPIMAGETOOL или добавьте appimagetool в PATH."
    )


def _resolve_executable(value: str) -> Path | None:
    if not _has_path_separator(value):
        found = shutil.which(value)
        if found:
            return Path(found)
    path = Path(value)
    if path.exists():
        return path
    return None


def _has_path_separator(value: str) -> bool:
    return os.sep in value or bool(os.altsep and os.altsep in value)


def _smoke_test_artifact(
    artifact_path: Path,
    *,
    manifest: dict[str, Any],
    platform_id: str,
    artifact: str,
) -> None:
    if artifact == "archive":
        _smoke_test_archive(artifact_path, manifest=manifest, platform_id=platform_id)
    elif artifact == "onefile":
        _smoke_test_onefile(artifact_path, manifest=manifest, platform_id=platform_id)
    elif artifact == "appimage":
        _smoke_test_appimage(artifact_path)
    else:
        raise RuntimeError(f"Неподдерживаемый smoke-test для artifact={artifact}")


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
    executable_name = (
        f"{app_name}.exe" if platform_id.startswith("windows-") else app_name
    )
    if not any(PurePosixPath(name).name == executable_name for name in members):
        raise RuntimeError(f"В архиве не найден исполняемый файл {executable_name}.")
    if not any(
        PurePosixPath(name).name == "oiv_ampelometric_scales.json" for name in members
    ):
        raise RuntimeError("В архиве не найден oiv_ampelometric_scales.json.")
    if not any(
        "magicborder/assets/" in name or "magicborder/assets\\" in name
        for name in members
    ):
        raise RuntimeError("В архиве не найдены assets MagicBorder.")


def _smoke_test_onefile(
    artifact_path: Path,
    *,
    manifest: dict[str, Any],
    platform_id: str,
) -> None:
    expected_name = _artifact_name(
        str(manifest["app"]["name"]),
        str(manifest["app"]["version"]),
        platform_id,
        "onefile",
    )
    if not artifact_path.exists() or artifact_path.stat().st_size <= 0:
        raise RuntimeError(f"Onefile exe не создан или пуст: {artifact_path}")
    if artifact_path.name != expected_name:
        raise RuntimeError(f"Onefile exe имеет неожиданное имя: {artifact_path.name}")
    if platform_id.startswith("windows-") and artifact_path.suffix.lower() != ".exe":
        raise RuntimeError(
            f"Windows onefile должен иметь расширение .exe: {artifact_path}"
        )


def _smoke_test_appimage(artifact_path: Path) -> None:
    if not artifact_path.exists() or artifact_path.stat().st_size <= 0:
        raise RuntimeError(f"AppImage не создан или пуст: {artifact_path}")
    if artifact_path.suffix != ".AppImage":
        raise RuntimeError(f"AppImage имеет неожиданное расширение: {artifact_path}")
    if not os.access(artifact_path, os.X_OK):
        raise RuntimeError(f"AppImage должен быть исполняемым: {artifact_path}")


def _archive_members(archive_path: Path) -> list[str]:
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            return archive.namelist()
    if archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            return archive.getnames()
    raise RuntimeError(f"Неподдерживаемый формат архива: {archive_path}")


def _artifact_output_path(
    *,
    manifest: dict[str, Any],
    platform_id: str,
    output_root: Path,
    artifact: str,
) -> Path:
    app = manifest["app"]
    return (
        output_root
        / platform_id
        / _artifact_name(str(app["name"]), str(app["version"]), platform_id, artifact)
    )


def _archive_name(app_name: str, version: str, platform_id: str) -> str:
    return _artifact_name(app_name, version, platform_id, "archive")


def _artifact_name(app_name: str, version: str, platform_id: str, artifact: str) -> str:
    if artifact == "archive":
        suffix = ".zip" if platform_id.startswith(("windows-", "macos-")) else ".tar.gz"
    elif artifact == "appimage":
        suffix = ".AppImage"
    elif artifact == "onefile":
        suffix = ".exe"
    else:
        raise ValueError(f"Неподдерживаемый тип артефакта: {artifact}")
    return f"{app_name}-{version}-{platform_id}{suffix}"


def _app_dir_name(manifest: dict[str, Any]) -> str:
    app = manifest["app"]
    display_name = str(app.get("display_name", app["name"]))
    safe_name = "".join(ch for ch in display_name if ch.isalnum() or ch in {"-", "_"})
    return f"{safe_name or app['name']}.AppDir"


def _appimage_categories(manifest: dict[str, Any]) -> str:
    categories = manifest.get("appimage", {}).get(
        "desktop_categories", ["Graphics", "Science", "Education"]
    )
    cleaned = [str(category).strip().strip(";") for category in categories]
    cleaned = [category for category in cleaned if category]
    if not cleaned:
        cleaned = ["Graphics"]
    return ";".join(cleaned) + ";"


def _appimage_icon_source(manifest: dict[str, Any]) -> Path:
    configured = str(manifest.get("appimage", {}).get("icon", "")).strip()
    if configured:
        return _repo_path(configured)

    default_icon = REPO_ROOT / "build_tools" / "assets" / "magicborder.svg"
    if default_icon.exists():
        return default_icon
    return REPO_ROOT / "src" / "magicborder" / "assets" / "icons" / "about.svg"


def _repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _add_executable_bits(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


if __name__ == "__main__":
    raise SystemExit(main())
