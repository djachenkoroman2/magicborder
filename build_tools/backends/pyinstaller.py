from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BundleMode = Literal["onedir", "onefile"]


@dataclass(frozen=True, slots=True)
class PyInstallerPlan:
    command: list[str]
    build_root: Path
    dist_path: Path
    launcher_path: Path
    module_name: str
    function_name: str
    app_name: str
    platform_id: str
    bundle_mode: BundleMode


def create_plan(
    *,
    repo_root: Path,
    manifest: dict,
    platform_id: str,
    clean: bool,
    bundle_mode: BundleMode = "onedir",
) -> PyInstallerPlan:
    if bundle_mode not in {"onedir", "onefile"}:
        raise ValueError(f"Неподдерживаемый режим PyInstaller: {bundle_mode}")

    app = manifest["app"]
    app_name = str(app["name"])
    build_root = repo_root / "build" / "portable" / platform_id
    dist_path = build_root / "pyinstaller-dist"
    work_path = build_root / "pyinstaller-work"
    spec_path = build_root / "pyinstaller-spec"
    launcher_path = build_root / f"_{app_name}_launcher.py"

    entrypoint = str(app["entrypoint"])
    module_name, function_name = _split_entrypoint(entrypoint)
    add_data_separator = ";" if platform_id.startswith("windows-") else ":"

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--name",
        app_name,
        "--distpath",
        str(dist_path),
        "--workpath",
        str(work_path),
        "--specpath",
        str(spec_path),
    ]
    if clean:
        command.append("--clean")
    if bundle_mode == "onefile":
        command.append("--onefile")
    if bool(app.get("gui", False)):
        command.append("--windowed")

    for item in manifest.get("data", []):
        source = repo_root / str(item["source"])
        target = str(item["target"])
        command.extend(["--add-data", f"{source}{add_data_separator}{target}"])

    command.append(str(launcher_path))
    return PyInstallerPlan(
        command=command,
        build_root=build_root,
        dist_path=dist_path,
        launcher_path=launcher_path,
        module_name=module_name,
        function_name=function_name,
        app_name=app_name,
        platform_id=platform_id,
        bundle_mode=bundle_mode,
    )


def build(plan: PyInstallerPlan, *, clean: bool) -> Path:
    if clean and plan.build_root.exists():
        shutil.rmtree(plan.build_root)
    plan.build_root.mkdir(parents=True, exist_ok=True)
    _write_launcher(plan.launcher_path, plan.module_name, plan.function_name)
    subprocess.run(plan.command, check=True)
    return _find_bundle(plan)


def _split_entrypoint(entrypoint: str) -> tuple[str, str]:
    if ":" not in entrypoint:
        raise ValueError("entrypoint должен иметь формат 'module:function'.")
    module_name, function_name = entrypoint.split(":", 1)
    module_name = module_name.strip()
    function_name = function_name.strip()
    if not module_name or not function_name:
        raise ValueError("entrypoint должен иметь формат 'module:function'.")
    return module_name, function_name


def _write_launcher(path: Path, module_name: str, function_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                f"from {module_name} import {function_name} as _magicborder_main",
                "",
                "",
                "if __name__ == '__main__':",
                "    raise SystemExit(_magicborder_main())",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _find_bundle(plan: PyInstallerPlan) -> Path:
    app_name = plan.app_name
    candidates = [
        plan.dist_path / app_name,
        plan.dist_path / f"{app_name}.exe",
        plan.dist_path / f"{app_name}.app",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    existing = sorted(plan.dist_path.iterdir()) if plan.dist_path.exists() else []
    if len(existing) == 1:
        return existing[0]
    raise FileNotFoundError(
        f"PyInstaller не создал ожидаемый bundle в {plan.dist_path}."
    )
