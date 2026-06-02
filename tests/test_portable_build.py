from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from build_tools import portable_build
from build_tools.backends import pyinstaller


@pytest.fixture()
def build_manifest(tmp_path: Path) -> dict:
    icon_path = tmp_path / "magicborder.svg"
    icon_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    return {
        "app": {
            "name": "magicborder",
            "display_name": "MagicBorder",
            "version": "0.1.0",
            "entrypoint": "magicborder.main:main",
            "gui": True,
        },
        "python": {"requires": ">=3.11,<3.14", "recommended": "3.12"},
        "build": {"backend": "pyinstaller", "output_root": "dist"},
        "appimage": {
            "desktop_categories": ["Graphics", "Science", "Education"],
            "icon": str(icon_path),
        },
        "data": [],
    }


def test_default_artifact_is_archive() -> None:
    args = portable_build._parse_args([])

    assert args.artifact == "archive"


@pytest.mark.parametrize(
    ("platform_id", "artifact", "expected"),
    [
        ("linux-x86_64", "archive", "magicborder-0.1.0-linux-x86_64.tar.gz"),
        ("windows-x86_64", "archive", "magicborder-0.1.0-windows-x86_64.zip"),
        ("macos-x86_64", "archive", "magicborder-0.1.0-macos-x86_64.zip"),
        ("macos-arm64", "archive", "magicborder-0.1.0-macos-arm64.zip"),
        ("linux-x86_64", "appimage", "magicborder-0.1.0-linux-x86_64.AppImage"),
        ("windows-x86_64", "onefile", "magicborder-0.1.0-windows-x86_64.exe"),
    ],
)
def test_artifact_names(platform_id: str, artifact: str, expected: str) -> None:
    assert portable_build._artifact_name("magicborder", "0.1.0", platform_id, artifact) == expected


@pytest.mark.parametrize(
    ("artifact", "platform_id"),
    [
        ("appimage", "windows-x86_64"),
        ("appimage", "macos-x86_64"),
        ("onefile", "linux-x86_64"),
        ("onefile", "macos-arm64"),
    ],
)
def test_incompatible_artifact_platform_pairs_are_rejected(artifact: str, platform_id: str) -> None:
    with pytest.raises(SystemExit, match=f"Artifact {artifact}"):
        portable_build._validate_artifact_platform(artifact, platform_id)


def test_foreign_windows_onefile_dry_run_shows_plan(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(portable_build, "_current_platform_id", lambda: "linux-x86_64")

    result = portable_build.main(["--platform", "windows-x86_64", "--artifact", "onefile", "--dry-run"])

    output = capsys.readouterr().out
    assert result == 0
    assert "WARNING:" in output
    assert "Artifact: onefile" in output
    assert "magicborder-0.1.0-windows-x86_64.exe" in output
    assert "--onefile" in output


def test_foreign_platform_real_build_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(portable_build, "_current_platform_id", lambda: "linux-x86_64")

    with pytest.raises(SystemExit, match="Portable-сборка должна выполняться на целевой ОС"):
        portable_build.main(["--platform", "windows-x86_64", "--artifact", "onefile"])


def test_pyinstaller_onefile_mode_adds_onefile_flag(build_manifest: dict, tmp_path: Path) -> None:
    plan = pyinstaller.create_plan(
        repo_root=tmp_path,
        manifest=build_manifest,
        platform_id="windows-x86_64",
        clean=True,
        bundle_mode="onefile",
    )

    assert plan.bundle_mode == "onefile"
    assert "--onefile" in plan.command
    assert "--windowed" in plan.command


def test_pyinstaller_onedir_modes_do_not_add_onefile_flag(build_manifest: dict, tmp_path: Path) -> None:
    plan = pyinstaller.create_plan(
        repo_root=tmp_path,
        manifest=build_manifest,
        platform_id="linux-x86_64",
        clean=False,
        bundle_mode="onedir",
    )

    assert plan.bundle_mode == "onedir"
    assert "--onefile" not in plan.command


def test_pyinstaller_find_bundle_accepts_windows_exe(build_manifest: dict, tmp_path: Path) -> None:
    plan = pyinstaller.create_plan(
        repo_root=tmp_path,
        manifest=build_manifest,
        platform_id="windows-x86_64",
        clean=False,
        bundle_mode="onefile",
    )
    exe_path = plan.dist_path / "magicborder.exe"
    exe_path.parent.mkdir(parents=True)
    exe_path.write_bytes(b"exe")

    assert pyinstaller._find_bundle(plan) == exe_path


def test_package_onefile_copies_versioned_exe(build_manifest: dict, tmp_path: Path) -> None:
    bundle_path = tmp_path / "magicborder.exe"
    bundle_path.write_bytes(b"exe")

    result = portable_build._package_onefile(
        bundle_path=bundle_path,
        manifest=build_manifest,
        platform_id="windows-x86_64",
        output_root=tmp_path / "dist",
    )

    assert result == tmp_path / "dist" / "windows-x86_64" / "magicborder-0.1.0-windows-x86_64.exe"
    assert result.read_bytes() == b"exe"


def test_prepare_app_dir_creates_apprun_desktop_icon_and_bundle(build_manifest: dict, tmp_path: Path) -> None:
    bundle_path = tmp_path / "pyinstaller-dist" / "magicborder"
    bundle_path.mkdir(parents=True)
    executable = bundle_path / "magicborder"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")

    app_dir = portable_build._prepare_app_dir(
        bundle_path=bundle_path,
        manifest=build_manifest,
        build_root=tmp_path / "build",
    )

    app_run = app_dir / "AppRun"
    desktop = app_dir / "magicborder.desktop"
    share_desktop = app_dir / "usr" / "share" / "applications" / "magicborder.desktop"
    share_icon = app_dir / "usr" / "share" / "icons" / "hicolor" / "scalable" / "apps" / "magicborder.svg"

    assert app_dir.name == "MagicBorder.AppDir"
    assert (app_dir / "usr" / "bin" / "magicborder" / "magicborder").exists()
    assert os.access(app_run, os.X_OK)
    assert 'exec "$HERE/usr/bin/magicborder/magicborder" "$@"' in app_run.read_text(encoding="utf-8")
    assert "Categories=Graphics;Science;Education;" in desktop.read_text(encoding="utf-8")
    assert share_desktop.read_text(encoding="utf-8") == desktop.read_text(encoding="utf-8")
    assert (app_dir / "magicborder.svg").exists()
    assert share_icon.exists()


def test_smoke_test_archive_checks_executable_oiv_and_assets(build_manifest: dict, tmp_path: Path) -> None:
    archive_path = tmp_path / "magicborder-0.1.0-linux-x86_64.tar.gz"
    executable = tmp_path / "magicborder"
    oiv_json = tmp_path / "oiv_ampelometric_scales.json"
    asset = tmp_path / "about.svg"
    executable.write_bytes(b"bin")
    oiv_json.write_text("{}", encoding="utf-8")
    asset.write_text("<svg/>", encoding="utf-8")

    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(executable, arcname="magicborder-0.1.0-linux-x86_64/magicborder/magicborder")
        archive.add(oiv_json, arcname="magicborder-0.1.0-linux-x86_64/magicborder/oiv_ampelometric_scales.json")
        archive.add(asset, arcname="magicborder-0.1.0-linux-x86_64/magicborder/magicborder/assets/about.svg")

    portable_build._smoke_test_artifact(
        archive_path,
        manifest=build_manifest,
        platform_id="linux-x86_64",
        artifact="archive",
    )


def test_smoke_test_onefile_checks_versioned_exe(build_manifest: dict, tmp_path: Path) -> None:
    exe_path = tmp_path / "magicborder-0.1.0-windows-x86_64.exe"
    exe_path.write_bytes(b"exe")

    portable_build._smoke_test_artifact(
        exe_path,
        manifest=build_manifest,
        platform_id="windows-x86_64",
        artifact="onefile",
    )


def test_smoke_test_appimage_checks_executable_file(tmp_path: Path) -> None:
    appimage_path = tmp_path / "magicborder-0.1.0-linux-x86_64.AppImage"
    appimage_path.write_bytes(b"appimage")
    portable_build._add_executable_bits(appimage_path)

    portable_build._smoke_test_artifact(
        appimage_path,
        manifest={},
        platform_id="linux-x86_64",
        artifact="appimage",
    )
