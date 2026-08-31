from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_tools import portable_build  # noqa: E402
from build_tools.backends import pyinstaller  # noqa: E402

VALID_MANIFEST = """
[app]
name = "magicborder"
display_name = "MagicBorder"
version = "0.1.0"
entrypoint = "magicborder.main:main"
gui = true

[python]
requires = ">=3.11,<3.14"

[build]
backend = "pyinstaller"
output_root = "dist"
"""


@pytest.fixture()
def manifest_dict(tmp_path: Path) -> dict[str, Any]:
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
        "python": {"requires": ">=3.11,<3.14"},
        "build": {"backend": "pyinstaller", "output_root": "dist"},
        "appimage": {
            "desktop_categories": ["Graphics", "Science", "Education"],
            "icon": str(icon_path),
        },
        "data": [],
    }


def _write_manifest(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "manifest.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadManifestSections:
    @pytest.mark.parametrize("section", ["app", "python", "build"])
    def test_missing_section_is_rejected(self, tmp_path: Path, section: str) -> None:
        text = "\n".join(
            block
            for block in VALID_MANIFEST.strip().split("\n\n")
            if not block.startswith(f"[{section}]")
        )

        with pytest.raises(ValueError, match=rf"секцию \[{section}\]"):
            portable_build._load_manifest(_write_manifest(tmp_path, text))

    @pytest.mark.parametrize("section", ["app", "python", "build"])
    def test_section_must_be_a_table(self, tmp_path: Path, section: str) -> None:
        text = VALID_MANIFEST.replace(
            f"[{section}]", f"{section} = 5\n[{section}_unused]", 1
        )

        with pytest.raises(ValueError, match=rf"секцию \[{section}\]"):
            portable_build._load_manifest(_write_manifest(tmp_path, text))

    @pytest.mark.parametrize("key", ["name", "version", "entrypoint"])
    def test_empty_app_keys_are_rejected(self, tmp_path: Path, key: str) -> None:
        text = VALID_MANIFEST.replace(f'{key} = "', f'{key} = "   " # ', 1)

        with pytest.raises(ValueError, match=rf"\[app\] должен содержать {key}"):
            portable_build._load_manifest(_write_manifest(tmp_path, text))

    def test_valid_manifest_is_loaded(self, tmp_path: Path) -> None:
        manifest = portable_build._load_manifest(
            _write_manifest(tmp_path, VALID_MANIFEST)
        )

        assert manifest["app"]["name"] == "magicborder"
        assert manifest["build"]["backend"] == "pyinstaller"

    def test_repository_manifest_is_valid(self) -> None:
        manifest = portable_build._load_manifest(portable_build.DEFAULT_MANIFEST)

        assert manifest["app"]["entrypoint"] == "magicborder.main:main"


class TestLoadManifestData:
    def test_data_must_be_a_list(self, tmp_path: Path) -> None:
        text = 'data = "src"\n' + VALID_MANIFEST

        with pytest.raises(ValueError, match="data должен быть списком"):
            portable_build._load_manifest(_write_manifest(tmp_path, text))

    def test_missing_source_is_reported(self, tmp_path: Path) -> None:
        text = (
            VALID_MANIFEST + '\n[[data]]\nsource = "нет/такого/файла"\ntarget = "."\n'
        )

        with pytest.raises(FileNotFoundError, match="Файл данных не найден"):
            portable_build._load_manifest(_write_manifest(tmp_path, text))

    def test_empty_target_is_rejected(self, tmp_path: Path) -> None:
        text = VALID_MANIFEST + '\n[[data]]\nsource = "pyproject.toml"\ntarget = "  "\n'

        with pytest.raises(ValueError, match="должен содержать target"):
            portable_build._load_manifest(_write_manifest(tmp_path, text))

    def test_valid_data_entry_passes(self, tmp_path: Path) -> None:
        text = VALID_MANIFEST + '\n[[data]]\nsource = "pyproject.toml"\ntarget = "."\n'

        manifest = portable_build._load_manifest(_write_manifest(tmp_path, text))

        assert manifest["data"][0]["target"] == "."


class TestLoadManifestAppImage:
    def test_appimage_must_be_a_table(self, tmp_path: Path) -> None:
        text = "appimage = 5\n" + VALID_MANIFEST

        with pytest.raises(ValueError, match=r"\[appimage\] должен быть секцией TOML"):
            portable_build._load_manifest(_write_manifest(tmp_path, text))

    def test_missing_icon_is_reported(self, tmp_path: Path) -> None:
        text = VALID_MANIFEST + '\n[appimage]\nicon = "нет/иконки.svg"\n'

        with pytest.raises(FileNotFoundError, match="Иконка AppImage не найдена"):
            portable_build._load_manifest(_write_manifest(tmp_path, text))

    def test_categories_must_be_a_list(self, tmp_path: Path) -> None:
        text = VALID_MANIFEST + '\n[appimage]\ndesktop_categories = "Graphics"\n'

        with pytest.raises(ValueError, match="desktop_categories должен быть списком"):
            portable_build._load_manifest(_write_manifest(tmp_path, text))

    def test_appimage_section_without_extras_is_accepted(self, tmp_path: Path) -> None:
        text = VALID_MANIFEST + "\n[appimage]\n"

        manifest = portable_build._load_manifest(_write_manifest(tmp_path, text))

        assert manifest["appimage"] == {}


class TestPackageArchive:
    def _bundle(self, tmp_path: Path, *, as_directory: bool) -> Path:
        bundle_path = tmp_path / "pyinstaller-dist" / "magicborder"
        if as_directory:
            bundle_path.mkdir(parents=True)
            (bundle_path / "magicborder").write_bytes(b"bin")
            (bundle_path / "assets").mkdir()
            (bundle_path / "assets" / "about.svg").write_text(
                "<svg/>", encoding="utf-8"
            )
        else:
            bundle_path.parent.mkdir(parents=True)
            bundle_path.write_bytes(b"bin")
        return bundle_path

    @pytest.mark.parametrize(
        ("platform_id", "expected_name"),
        [
            ("windows-x86_64", "magicborder-0.1.0-windows-x86_64.zip"),
            ("macos-arm64", "magicborder-0.1.0-macos-arm64.zip"),
            ("linux-x86_64", "magicborder-0.1.0-linux-x86_64.tar.gz"),
        ],
    )
    def test_archive_format_per_platform(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
        platform_id: str,
        expected_name: str,
    ) -> None:
        bundle_path = self._bundle(tmp_path, as_directory=True)

        archive_path = portable_build._package_archive(
            bundle_path=bundle_path,
            manifest=manifest_dict,
            platform_id=platform_id,
            output_root=tmp_path / "dist",
            build_root=tmp_path / "build",
        )

        assert archive_path.name == expected_name
        assert archive_path.parent.name == platform_id
        members = portable_build._archive_members(archive_path)
        assert any(name.endswith("magicborder/magicborder") for name in members)

    def test_stale_package_directory_is_removed(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        build_root = tmp_path / "build"
        stale = (
            build_root / "package" / "magicborder-0.1.0-linux-x86_64" / "устаревшее.txt"
        )
        stale.parent.mkdir(parents=True)
        stale.write_text("старое", encoding="utf-8")
        bundle_path = self._bundle(tmp_path, as_directory=True)

        archive_path = portable_build._package_archive(
            bundle_path=bundle_path,
            manifest=manifest_dict,
            platform_id="linux-x86_64",
            output_root=tmp_path / "dist",
            build_root=build_root,
        )

        members = portable_build._archive_members(archive_path)
        assert not any("устаревшее.txt" in name for name in members)

    def test_onedir_bundle_is_copied_as_tree(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        bundle_path = self._bundle(tmp_path, as_directory=True)

        archive_path = portable_build._package_archive(
            bundle_path=bundle_path,
            manifest=manifest_dict,
            platform_id="linux-x86_64",
            output_root=tmp_path / "dist",
            build_root=tmp_path / "build",
        )

        members = portable_build._archive_members(archive_path)
        assert any(name.endswith("magicborder/assets/about.svg") for name in members)

    def test_single_file_bundle_is_copied_as_file(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        bundle_path = self._bundle(tmp_path, as_directory=False)

        archive_path = portable_build._package_archive(
            bundle_path=bundle_path,
            manifest=manifest_dict,
            platform_id="windows-x86_64",
            output_root=tmp_path / "dist",
            build_root=tmp_path / "build",
        )

        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
        assert "magicborder-0.1.0-windows-x86_64/magicborder" in names


class TestPackageAppImage:
    def _stub_subprocess(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        create_output: bool = True,
    ) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def fake_run(command, check, env):  # noqa: ANN001 - подпись повторяет вызов
            calls.append({"command": command, "check": check, "env": env})
            if create_output:
                Path(command[2]).write_bytes(b"appimage")
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(portable_build.subprocess, "run", fake_run)
        return calls

    def _bundle(self, tmp_path: Path) -> Path:
        bundle_path = tmp_path / "pyinstaller-dist" / "magicborder"
        bundle_path.mkdir(parents=True)
        (bundle_path / "magicborder").write_bytes(b"bin")
        return bundle_path

    def test_non_linux_platform_is_rejected(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        with pytest.raises(
            RuntimeError, match="AppImage поддержан только для linux-x86_64"
        ):
            portable_build._package_appimage(
                bundle_path=tmp_path,
                manifest=manifest_dict,
                platform_id="macos-arm64",
                output_root=tmp_path / "dist",
                build_root=tmp_path / "build",
                appimagetool=tmp_path / "appimagetool",
            )

    def test_existing_output_is_removed_first(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        output_root = tmp_path / "dist"
        output_path = (
            output_root / "linux-x86_64" / "magicborder-0.1.0-linux-x86_64.AppImage"
        )
        output_path.parent.mkdir(parents=True)
        output_path.write_bytes(b"old artifact")
        self._stub_subprocess(monkeypatch)

        result = portable_build._package_appimage(
            bundle_path=self._bundle(tmp_path),
            manifest=manifest_dict,
            platform_id="linux-x86_64",
            output_root=output_root,
            build_root=tmp_path / "build",
            appimagetool=tmp_path / "appimagetool",
        )

        assert result.read_bytes() == b"appimage"

    def test_arch_defaults_to_x86_64(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("ARCH", raising=False)
        calls = self._stub_subprocess(monkeypatch)

        portable_build._package_appimage(
            bundle_path=self._bundle(tmp_path),
            manifest=manifest_dict,
            platform_id="linux-x86_64",
            output_root=tmp_path / "dist",
            build_root=tmp_path / "build",
            appimagetool=tmp_path / "appimagetool",
        )

        assert calls[0]["env"]["ARCH"] == "x86_64"
        assert calls[0]["check"] is True

    def test_existing_arch_is_preserved(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ARCH", "aarch64")
        calls = self._stub_subprocess(monkeypatch)

        portable_build._package_appimage(
            bundle_path=self._bundle(tmp_path),
            manifest=manifest_dict,
            platform_id="linux-x86_64",
            output_root=tmp_path / "dist",
            build_root=tmp_path / "build",
            appimagetool=tmp_path / "appimagetool",
        )

        assert calls[0]["env"]["ARCH"] == "aarch64"

    def test_missing_output_after_tool_run_is_reported(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_subprocess(monkeypatch, create_output=False)

        with pytest.raises(RuntimeError, match="appimagetool не создал ожидаемый файл"):
            portable_build._package_appimage(
                bundle_path=self._bundle(tmp_path),
                manifest=manifest_dict,
                platform_id="linux-x86_64",
                output_root=tmp_path / "dist",
                build_root=tmp_path / "build",
                appimagetool=tmp_path / "appimagetool",
            )

    def test_result_gets_executable_bits(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stub_subprocess(monkeypatch)

        result = portable_build._package_appimage(
            bundle_path=self._bundle(tmp_path),
            manifest=manifest_dict,
            platform_id="linux-x86_64",
            output_root=tmp_path / "dist",
            build_root=tmp_path / "build",
            appimagetool=tmp_path / "appimagetool",
        )

        assert os.access(result, os.X_OK)


class TestResolveAppImageTool:
    def _tool(self, tmp_path: Path, name: str = "appimagetool") -> Path:
        tool = tmp_path / name
        tool.write_text("#!/bin/sh\n", encoding="utf-8")
        portable_build._add_executable_bits(tool)
        return tool

    def test_explicit_path_argument(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("APPIMAGETOOL", raising=False)
        tool = self._tool(tmp_path)

        assert portable_build._resolve_appimagetool(str(tool)) == tool

    def test_environment_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tool = self._tool(tmp_path)
        monkeypatch.setenv("APPIMAGETOOL", str(tool))

        assert portable_build._resolve_appimagetool(None) == tool

    def test_path_lookup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("APPIMAGETOOL", raising=False)
        tool = self._tool(tmp_path)
        monkeypatch.setattr(
            portable_build.shutil,
            "which",
            lambda name: str(tool) if name == "appimagetool" else None,
        )

        assert portable_build._resolve_appimagetool(None) == tool

    def test_not_found_anywhere(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("APPIMAGETOOL", raising=False)
        monkeypatch.setattr(portable_build.shutil, "which", lambda _name: None)

        with pytest.raises(RuntimeError, match="appimagetool не найден"):
            portable_build._resolve_appimagetool(None)

    def test_bad_explicit_path_never_falls_back(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # `raise` стоит внутри цикла, поэтому неверный --appimagetool сразу
        # обрывает поиск: ни APPIMAGETOOL, ни PATH уже не проверяются.
        fallback = self._tool(tmp_path, name="fallback-appimagetool")
        monkeypatch.setenv("APPIMAGETOOL", str(fallback))
        monkeypatch.setattr(portable_build.shutil, "which", lambda _name: str(fallback))

        with pytest.raises(RuntimeError, match="appimagetool не найден"):
            portable_build._resolve_appimagetool(str(tmp_path / "нет-такого-файла"))


class TestResolveExecutable:
    def test_bare_name_is_looked_up_in_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            portable_build.shutil, "which", lambda _name: "/usr/bin/tool"
        )

        assert portable_build._resolve_executable("tool") == Path("/usr/bin/tool")

    def test_path_with_separator_is_checked_directly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            portable_build.shutil,
            "which",
            lambda name: calls.append(name) or None,
        )
        tool = tmp_path / "tool"
        tool.write_text("#!/bin/sh\n", encoding="utf-8")

        assert portable_build._resolve_executable(str(tool)) == tool
        assert calls == []

    def test_missing_value_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(portable_build.shutil, "which", lambda _name: None)

        assert portable_build._resolve_executable("нет-такой-программы") is None


class TestCurrentPlatformId:
    @pytest.mark.parametrize(
        ("system", "machine", "expected"),
        [
            ("Linux", "x86_64", "linux-x86_64"),
            ("Linux", "AMD64", "linux-x86_64"),
            ("Windows", "AMD64", "windows-x86_64"),
            ("Darwin", "x86_64", "macos-x86_64"),
            ("Darwin", "arm64", "macos-arm64"),
            ("Darwin", "aarch64", "macos-arm64"),
        ],
    )
    def test_supported_pairs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        system: str,
        machine: str,
        expected: str,
    ) -> None:
        monkeypatch.setattr(portable_build.platform, "system", lambda: system)
        monkeypatch.setattr(portable_build.platform, "machine", lambda: machine)

        assert portable_build._current_platform_id() == expected

    @pytest.mark.parametrize(
        ("system", "machine"),
        [("Linux", "armv7l"), ("FreeBSD", "x86_64"), ("Windows", "arm64")],
    )
    def test_unsupported_pairs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        system: str,
        machine: str,
    ) -> None:
        monkeypatch.setattr(portable_build.platform, "system", lambda: system)
        monkeypatch.setattr(portable_build.platform, "machine", lambda: machine)

        with pytest.raises(SystemExit, match="Текущая платформа не поддержана"):
            portable_build._current_platform_id()


class TestParseArgs:
    def test_defaults(self) -> None:
        args = portable_build._parse_args([])

        assert args.manifest == str(portable_build.DEFAULT_MANIFEST)
        assert args.backend is None
        assert args.platform == "auto"
        assert args.artifact == "archive"
        assert args.appimagetool is None
        assert args.output_dir is None
        assert args.clean is False
        assert args.skip_smoke_test is False
        assert args.dry_run is False

    @pytest.mark.parametrize(
        "argv",
        [
            ["--platform", "solaris-sparc"],
            ["--artifact", "msi"],
            ["--backend", "nuitka"],
        ],
    )
    def test_invalid_choices_are_rejected(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit):
            portable_build._parse_args(argv)


class _FakePlan:
    def __init__(self, build_root: Path) -> None:
        self.command = ["pyinstaller", "--noconfirm"]
        self.build_root = build_root


class _FakeBackend:
    def __init__(self, build_root: Path) -> None:
        self.build_root = build_root
        self.create_plan_kwargs: dict[str, Any] = {}
        self.build_calls: list[bool] = []

    def create_plan(self, **kwargs: Any) -> _FakePlan:
        self.create_plan_kwargs = kwargs
        return _FakePlan(self.build_root)

    def build(self, plan: _FakePlan, *, clean: bool) -> Path:
        self.build_calls.append(clean)
        bundle_path = self.build_root / "bundle"
        bundle_path.mkdir(parents=True, exist_ok=True)
        return bundle_path


@pytest.fixture()
def mocked_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    backend = _FakeBackend(tmp_path / "build")
    packaged: list[dict[str, Any]] = []
    smoke_tested: list[dict[str, Any]] = []

    def fake_package_archive(**kwargs: Any) -> Path:
        packaged.append(kwargs)
        artifact = tmp_path / "artifact.tar.gz"
        artifact.write_bytes(b"artifact")
        return artifact

    monkeypatch.setattr(portable_build, "_load_pyinstaller_backend", lambda: backend)
    monkeypatch.setattr(portable_build, "_package_archive", fake_package_archive)
    monkeypatch.setattr(
        portable_build,
        "_smoke_test_artifact",
        lambda path, **kwargs: smoke_tested.append({"path": path, **kwargs}),
    )
    monkeypatch.setattr(portable_build, "_current_platform_id", lambda: "linux-x86_64")
    return {"backend": backend, "packaged": packaged, "smoke_tested": smoke_tested}


class TestMain:
    def test_clean_flag_is_forwarded(self, mocked_build: dict[str, Any]) -> None:
        assert portable_build.main(["--platform", "linux-x86_64", "--clean"]) == 0

        assert mocked_build["backend"].create_plan_kwargs["clean"] is True
        assert mocked_build["backend"].build_calls == [True]

    def test_smoke_test_runs_by_default(self, mocked_build: dict[str, Any]) -> None:
        portable_build.main(["--platform", "linux-x86_64"])

        assert len(mocked_build["smoke_tested"]) == 1
        assert mocked_build["smoke_tested"][0]["artifact"] == "archive"

    def test_skip_smoke_test_flag(self, mocked_build: dict[str, Any]) -> None:
        portable_build.main(["--platform", "linux-x86_64", "--skip-smoke-test"])

        assert mocked_build["smoke_tested"] == []

    def test_output_dir_overrides_manifest_output_root(
        self,
        mocked_build: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "готовые-сборки"

        portable_build.main(
            ["--platform", "linux-x86_64", "--output-dir", str(output_dir)]
        )

        assert mocked_build["packaged"][0]["output_root"] == output_dir

    def test_manifest_output_root_is_resolved_from_repo_root(
        self,
        mocked_build: dict[str, Any],
    ) -> None:
        portable_build.main(["--platform", "linux-x86_64"])

        assert (
            mocked_build["packaged"][0]["output_root"]
            == portable_build.REPO_ROOT / "dist"
        )

    def test_unsupported_manifest_backend(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            portable_build, "_current_platform_id", lambda: "linux-x86_64"
        )
        manifest_path = _write_manifest(
            tmp_path,
            VALID_MANIFEST.replace('backend = "pyinstaller"', 'backend = "nuitka"'),
        )

        with pytest.raises(SystemExit, match="Неподдерживаемый backend: nuitka"):
            portable_build.main(["--manifest", str(manifest_path), "--dry-run"])

    def test_relative_manifest_is_resolved_from_repo_root(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            portable_build, "_current_platform_id", lambda: "linux-x86_64"
        )
        monkeypatch.chdir(tmp_path)

        result = portable_build.main(
            ["--manifest", "build_tools/manifests/magicborder.toml", "--dry-run"]
        )

        assert result == 0
        assert "App: MagicBorder 0.1.0" in capsys.readouterr().out


class TestAppDirContents:
    def test_desktop_categories_come_from_manifest(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        manifest_dict["appimage"]["desktop_categories"] = ["Utility", "Development;"]

        portable_build._write_desktop_entries(tmp_path, manifest=manifest_dict)

        desktop_text = (tmp_path / "magicborder.desktop").read_text(encoding="utf-8")
        assert "Categories=Utility;Development;" in desktop_text
        assert "Name=MagicBorder" in desktop_text

    @pytest.mark.parametrize("categories", [[], ["  ", ";"]])
    def test_blank_categories_fall_back_to_graphics(
        self,
        manifest_dict: dict[str, Any],
        categories: list[str],
    ) -> None:
        manifest_dict["appimage"]["desktop_categories"] = categories

        assert portable_build._appimage_categories(manifest_dict) == "Graphics;"

    def test_default_categories_without_appimage_section(
        self,
        manifest_dict: dict[str, Any],
    ) -> None:
        manifest_dict.pop("appimage")

        assert (
            portable_build._appimage_categories(manifest_dict)
            == "Graphics;Science;Education;"
        )

    def test_icon_falls_back_to_repository_asset(
        self, manifest_dict: dict[str, Any]
    ) -> None:
        manifest_dict.pop("appimage")

        icon_source = portable_build._appimage_icon_source(manifest_dict)

        assert (
            icon_source
            == portable_build.REPO_ROOT / "build_tools" / "assets" / "magicborder.svg"
        )
        assert icon_source.exists()

    def test_copy_icon_without_manifest_entry(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        manifest_dict["appimage"] = {}
        app_dir = tmp_path / "MagicBorder.AppDir"
        app_dir.mkdir()

        portable_build._copy_appimage_icon(app_dir, manifest=manifest_dict)

        assert (app_dir / "magicborder.svg").is_file()
        assert (
            app_dir
            / "usr"
            / "share"
            / "icons"
            / "hicolor"
            / "scalable"
            / "apps"
            / "magicborder.svg"
        ).is_file()

    def test_apprun_content_and_permissions(self, tmp_path: Path) -> None:
        app_run = tmp_path / "AppRun"

        portable_build._write_apprun(
            app_run, app_name="magicborder", executable_name="magicborder"
        )

        text = app_run.read_text(encoding="utf-8")
        assert text.startswith("#!/bin/sh\n")
        assert 'HERE="$(dirname "$(readlink -f "$0")")"' in text
        assert 'exec "$HERE/usr/bin/magicborder/magicborder" "$@"' in text
        assert text.endswith("\n")
        assert os.access(app_run, os.X_OK)


def _make_tar(path: Path, members: dict[str, bytes], tmp_path: Path) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            source = tmp_path / "staging" / name.replace("/", "_")
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(payload)
            archive.add(source, arcname=name)
    return path


class TestSmokeTests:
    def _archive_members(
        self, prefix: str = "magicborder-0.1.0-linux-x86_64"
    ) -> dict[str, bytes]:
        return {
            f"{prefix}/magicborder/magicborder": b"bin",
            f"{prefix}/magicborder/oiv_ampelometric_scales.json": b"{}",
            f"{prefix}/magicborder/magicborder/assets/about.svg": b"<svg/>",
        }

    def test_missing_archive_is_reported(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        with pytest.raises(RuntimeError, match="Архив не создан или пуст"):
            portable_build._smoke_test_archive(
                tmp_path / "нет.tar.gz",
                manifest=manifest_dict,
                platform_id="linux-x86_64",
            )

    def test_archive_without_executable_is_reported(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        members = self._archive_members()
        members.pop("magicborder-0.1.0-linux-x86_64/magicborder/magicborder")
        archive_path = _make_tar(tmp_path / "a.tar.gz", members, tmp_path)

        with pytest.raises(
            RuntimeError, match="не найден исполняемый файл magicborder"
        ):
            portable_build._smoke_test_archive(
                archive_path,
                manifest=manifest_dict,
                platform_id="linux-x86_64",
            )

    def test_archive_without_oiv_json_is_reported(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        members = self._archive_members()
        members.pop(
            "magicborder-0.1.0-linux-x86_64/magicborder/oiv_ampelometric_scales.json"
        )
        archive_path = _make_tar(tmp_path / "a.tar.gz", members, tmp_path)

        with pytest.raises(RuntimeError, match="oiv_ampelometric_scales.json"):
            portable_build._smoke_test_archive(
                archive_path,
                manifest=manifest_dict,
                platform_id="linux-x86_64",
            )

    def test_archive_without_assets_is_reported(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        members = self._archive_members()
        members.pop(
            "magicborder-0.1.0-linux-x86_64/magicborder/magicborder/assets/about.svg"
        )
        archive_path = _make_tar(tmp_path / "a.tar.gz", members, tmp_path)

        with pytest.raises(RuntimeError, match="не найдены assets MagicBorder"):
            portable_build._smoke_test_archive(
                archive_path,
                manifest=manifest_dict,
                platform_id="linux-x86_64",
            )

    def test_windows_archive_expects_exe(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        archive_path = tmp_path / "magicborder-0.1.0-windows-x86_64.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name, payload in self._archive_members(
                "magicborder-0.1.0-windows-x86_64"
            ).items():
                archive.writestr(name, payload)

        with pytest.raises(RuntimeError, match="magicborder.exe"):
            portable_build._smoke_test_archive(
                archive_path,
                manifest=manifest_dict,
                platform_id="windows-x86_64",
            )

    def test_onefile_with_unexpected_name_is_reported(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        artifact = tmp_path / "magicborder.exe"
        artifact.write_bytes(b"exe")

        with pytest.raises(RuntimeError, match="неожиданное имя"):
            portable_build._smoke_test_onefile(
                artifact,
                manifest=manifest_dict,
                platform_id="windows-x86_64",
            )

    def test_missing_onefile_is_reported(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        with pytest.raises(RuntimeError, match="Onefile exe не создан или пуст"):
            portable_build._smoke_test_onefile(
                tmp_path / "magicborder-0.1.0-windows-x86_64.exe",
                manifest=manifest_dict,
                platform_id="windows-x86_64",
            )

    def test_missing_appimage_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="AppImage не создан или пуст"):
            portable_build._smoke_test_appimage(tmp_path / "нет.AppImage")

    def test_appimage_with_wrong_suffix_is_reported(self, tmp_path: Path) -> None:
        artifact = tmp_path / "magicborder.bin"
        artifact.write_bytes(b"appimage")

        with pytest.raises(RuntimeError, match="неожиданное расширение"):
            portable_build._smoke_test_appimage(artifact)

    def test_non_executable_appimage_is_reported(self, tmp_path: Path) -> None:
        artifact = tmp_path / "magicborder.AppImage"
        artifact.write_bytes(b"appimage")
        artifact.chmod(0o644)

        with pytest.raises(RuntimeError, match="должен быть исполняемым"):
            portable_build._smoke_test_appimage(artifact)

    def test_unsupported_artifact_kind(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Неподдерживаемый smoke-test"):
            portable_build._smoke_test_artifact(
                tmp_path / "a.bin",
                manifest={},
                platform_id="linux-x86_64",
                artifact="msi",
            )


class TestArchiveMembers:
    def test_zip_members(self, tmp_path: Path) -> None:
        archive_path = tmp_path / "a.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("root/file.txt", "данные")

        assert portable_build._archive_members(archive_path) == ["root/file.txt"]

    def test_tar_gz_members(self, tmp_path: Path) -> None:
        archive_path = _make_tar(
            tmp_path / "a.tar.gz", {"root/file.txt": b"data"}, tmp_path
        )

        assert portable_build._archive_members(archive_path) == ["root/file.txt"]

    def test_unsupported_format(self, tmp_path: Path) -> None:
        archive_path = tmp_path / "a.7z"
        archive_path.write_bytes(b"data")

        with pytest.raises(RuntimeError, match="Неподдерживаемый формат архива"):
            portable_build._archive_members(archive_path)


class TestArtifactName:
    def test_unsupported_artifact_kind(self) -> None:
        with pytest.raises(ValueError, match="Неподдерживаемый тип артефакта"):
            portable_build._artifact_name("magicborder", "0.1.0", "linux-x86_64", "msi")

    def test_archive_name_helper_matches_artifact_name(self) -> None:
        assert portable_build._archive_name("magicborder", "0.1.0", "linux-x86_64") == (
            "magicborder-0.1.0-linux-x86_64.tar.gz"
        )


class TestPyInstallerBackend:
    def test_find_bundle_reports_missing_output(
        self, manifest_dict: dict[str, Any], tmp_path: Path
    ) -> None:
        plan = pyinstaller.create_plan(
            repo_root=tmp_path,
            manifest=manifest_dict,
            platform_id="linux-x86_64",
            clean=False,
        )

        with pytest.raises(FileNotFoundError, match="не создал ожидаемый bundle"):
            pyinstaller._find_bundle(plan)

    def test_find_bundle_accepts_single_unexpected_entry(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        plan = pyinstaller.create_plan(
            repo_root=tmp_path,
            manifest=manifest_dict,
            platform_id="linux-x86_64",
            clean=False,
        )
        plan.dist_path.mkdir(parents=True)
        unexpected = plan.dist_path / "portable-bundle"
        unexpected.write_bytes(b"bin")

        assert pyinstaller._find_bundle(plan) == unexpected

    @pytest.mark.parametrize(
        ("platform_id", "separator"),
        [
            ("windows-x86_64", ";"),
            ("linux-x86_64", ":"),
            ("macos-arm64", ":"),
            ("macos-x86_64", ":"),
        ],
    )
    def test_add_data_separator_per_platform(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
        platform_id: str,
        separator: str,
    ) -> None:
        manifest_dict["data"] = [
            {"source": "src/magicborder/assets", "target": "magicborder/assets"}
        ]

        plan = pyinstaller.create_plan(
            repo_root=tmp_path,
            manifest=manifest_dict,
            platform_id=platform_id,
            clean=False,
        )

        add_data_value = plan.command[plan.command.index("--add-data") + 1]
        assert add_data_value.endswith(f"{separator}magicborder/assets")
        assert add_data_value.startswith(
            str(tmp_path / "src" / "magicborder" / "assets")
        )

    def test_hidden_imports_and_excludes_are_not_forwarded_yet(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        # Манифест может объявить эти ключи, но create_plan их пока не читает.
        manifest_dict["build"]["hidden_imports"] = ["magicborder.detector"]
        manifest_dict["build"]["excludes"] = ["tkinter"]

        plan = pyinstaller.create_plan(
            repo_root=tmp_path,
            manifest=manifest_dict,
            platform_id="linux-x86_64",
            clean=False,
        )

        assert "--hidden-import" not in plan.command
        assert "--exclude-module" not in plan.command

    @pytest.mark.parametrize("bundle_mode", ["onedir-plus", "", "ONEFILE", None])
    def test_invalid_bundle_mode(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
        bundle_mode: Any,
    ) -> None:
        with pytest.raises(ValueError, match="Неподдерживаемый режим PyInstaller"):
            pyinstaller.create_plan(
                repo_root=tmp_path,
                manifest=manifest_dict,
                platform_id="linux-x86_64",
                clean=False,
                bundle_mode=bundle_mode,
            )

    def test_plan_paths_and_names(
        self, manifest_dict: dict[str, Any], tmp_path: Path
    ) -> None:
        plan = pyinstaller.create_plan(
            repo_root=tmp_path,
            manifest=manifest_dict,
            platform_id="linux-x86_64",
            clean=True,
        )

        assert plan.build_root == tmp_path / "build" / "portable" / "linux-x86_64"
        assert plan.dist_path == plan.build_root / "pyinstaller-dist"
        assert plan.launcher_path == plan.build_root / "_magicborder_launcher.py"
        assert plan.module_name == "magicborder.main"
        assert plan.function_name == "main"
        assert "--clean" in plan.command
        assert plan.command[-1] == str(plan.launcher_path)

    @pytest.mark.parametrize(
        ("entrypoint", "expected"),
        [
            ("magicborder.main:main", ("magicborder.main", "main")),
            (" pkg.mod : run ", ("pkg.mod", "run")),
            ("a:b:c", ("a", "b:c")),
        ],
    )
    def test_split_entrypoint_valid(
        self, entrypoint: str, expected: tuple[str, str]
    ) -> None:
        assert pyinstaller._split_entrypoint(entrypoint) == expected

    @pytest.mark.parametrize(
        "entrypoint", ["magicborder.main", ":main", "magicborder.main:", "  :  "]
    )
    def test_split_entrypoint_invalid(self, entrypoint: str) -> None:
        with pytest.raises(ValueError, match="module:function"):
            pyinstaller._split_entrypoint(entrypoint)

    def test_write_launcher_content(self, tmp_path: Path) -> None:
        launcher_path = tmp_path / "nested" / "_launcher.py"

        pyinstaller._write_launcher(launcher_path, "magicborder.main", "main")

        text = launcher_path.read_text(encoding="utf-8")
        assert "from magicborder.main import main as _magicborder_main" in text
        assert "raise SystemExit(_magicborder_main())" in text

    def test_build_runs_command_and_returns_bundle(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = pyinstaller.create_plan(
            repo_root=tmp_path,
            manifest=manifest_dict,
            platform_id="linux-x86_64",
            clean=False,
        )
        commands: list[list[str]] = []

        def fake_run(command, check):  # noqa: ARG001 - подпись повторяет вызов
            commands.append(command)
            bundle = plan.dist_path / "magicborder"
            bundle.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(pyinstaller.subprocess, "run", fake_run)

        bundle_path = pyinstaller.build(plan, clean=False)

        assert commands == [plan.command]
        assert bundle_path == plan.dist_path / "magicborder"
        assert plan.launcher_path.is_file()

    def test_build_with_clean_removes_build_root(
        self,
        manifest_dict: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = pyinstaller.create_plan(
            repo_root=tmp_path,
            manifest=manifest_dict,
            platform_id="linux-x86_64",
            clean=True,
        )
        stale = plan.build_root / "старое.txt"
        stale.parent.mkdir(parents=True)
        stale.write_text("старое", encoding="utf-8")

        def fake_run(command, check):  # noqa: ARG001 - подпись повторяет вызов
            (plan.dist_path / "magicborder").mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(pyinstaller.subprocess, "run", fake_run)

        pyinstaller.build(plan, clean=True)

        assert not stale.exists()


def test_shutil_is_available_for_helpers() -> None:
    assert shutil.which is not None
