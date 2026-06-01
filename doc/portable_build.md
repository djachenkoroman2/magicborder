# Сборка portable-приложений MagicBorder

Этот документ описывает первый универсальный механизм portable-сборки MagicBorder.
Универсальным является сценарий сборки, а не итоговый бинарник: сборка для
Windows выполняется на Windows, для Linux - на Linux, для macOS - на macOS.

Матрица зависимостей и ограничения по wheel-покрытию описаны в
`doc/portable_dependency_matrix.md`.

## Поддерживаемые цели

На первом этапе в сборочном wrapper-е заведены цели:

| Цель | Где собирать | Формат результата |
|---|---|---|
| `linux-x86_64` | Linux x86_64 | `.tar.gz` |
| `windows-x86_64` | Windows x86_64 | `.zip` |
| `macos-x86_64` | macOS x86_64 | `.zip` |
| `macos-arm64` | macOS arm64 | `.zip` |

`linux-aarch64` пока не считается готовой portable-целью: у Qt-стека
`PyQt5`/`PyQt5-Qt5` нет нужного wheel-покрытия для выбранной связки.

## Подготовка окружения

Рекомендуемый Python для первых сборок: CPython `3.12.x`.
Проектный диапазон: `>=3.11,<3.14`.

Перед сборкой:

```bash
uv sync --frozen --extra build
```

Проверить зависимости и импорты:

```bash
uv lock --check
uv run --extra build python -c "import PyQt5, cv2, numpy, PIL"
```

## Dry-run

Dry-run показывает план, backend-команду и будущий путь артефакта, но не запускает
PyInstaller:

```bash
uv run --extra build python build_tools/portable_build.py --dry-run
```

Можно посмотреть план для другой платформы:

```bash
uv run --extra build python build_tools/portable_build.py --platform windows-x86_64 --dry-run
```

Фактическая сборка чужой платформы намеренно запрещена: wrapper не пытается
кросс-компилировать.

## Сборка текущей платформы

```bash
uv run --extra build python build_tools/portable_build.py --clean
```

По умолчанию используется manifest:

```text
build_tools/manifests/magicborder.toml
```

Можно указать manifest и выходной каталог явно:

```bash
uv run --extra build python build_tools/portable_build.py \
  --manifest build_tools/manifests/magicborder.toml \
  --output-dir dist \
  --clean
```

## Результат

Артефакты складываются в каталог текущей платформы:

```text
dist/
  linux-x86_64/
    magicborder-0.1.0-linux-x86_64.tar.gz
  windows-x86_64/
    magicborder-0.1.0-windows-x86_64.zip
  macos-x86_64/
    magicborder-0.1.0-macos-x86_64.zip
  macos-arm64/
    magicborder-0.1.0-macos-arm64.zip
```

Внутри архива находится одна корневая папка:

```text
magicborder-0.1.0-<platform>/
```

В сборку включаются:

- исполняемый bundle PyInstaller;
- `oiv_ampelometric_scales.json`;
- `magicborder/assets`.

После сборки wrapper выполняет smoke-test архива:

- архив существует и не пустой;
- внутри есть исполняемый файл;
- внутри есть OIV JSON;
- внутри есть assets.

Если smoke-test нужно временно отключить:

```bash
uv run --extra build python build_tools/portable_build.py --skip-smoke-test
```

## Платформенные замечания

### Linux

PyInstaller не упаковывает `glibc`; итоговый бинарник зависит от версии `glibc`
на системе сборки. Поэтому Linux portable-артефакт нужно собирать на самой старой
Linux-системе или в самом старом контейнере, который планируется поддерживать.
Сборка на новой системе может не запускаться на более старой.

### Windows

Windows-сборку нужно выполнять на Windows x86_64. В проекте для Windows
используется `PyQt5-Qt5==5.15.2`, потому что свежая ветка `PyQt5-Qt5 5.15.19`
не публикует Windows wheels.

### macOS

macOS-сборки нужно выполнять на macOS. Для x86_64 и arm64 предполагаются отдельные
артефакты. На этапе первого wrapper-а universal2-сборка не настраивается.

## Служебные файлы

Сборочный механизм состоит из:

```text
build_tools/
  portable_build.py
  backends/
    pyinstaller.py
  manifests/
    magicborder.toml
```

`build_tools/portable_build.py` отвечает за CLI, platform detection, упаковку
архива и smoke-test. `build_tools/backends/pyinstaller.py` содержит только
backend-специфичную команду PyInstaller.

Реальные build-output файлы игнорируются git. В репозитории оставлен только
`dist/.gitkeep`, чтобы каталог назначения существовал.
