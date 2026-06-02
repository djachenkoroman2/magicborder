# Сборка portable-приложений MagicBorder

Этот документ описывает первый универсальный механизм portable-сборки MagicBorder.
Универсальным является сценарий сборки, а не итоговый бинарник: сборка для
Windows выполняется на Windows, для Linux - на Linux, для macOS - на macOS.

Матрица зависимостей и ограничения по wheel-покрытию описаны в
`doc/portable_dependency_matrix.md`.

## Поддерживаемые цели

На первом этапе в сборочном wrapper-е заведены цели:

| Цель | Где собирать | Формат по умолчанию | Опциональные форматы |
|---|---|---|---|
| `linux-x86_64` | Linux x86_64 | `.tar.gz` | `.AppImage` |
| `windows-x86_64` | Windows x86_64 | `.zip` | single-file `.exe` |
| `macos-x86_64` | macOS x86_64 | `.zip` | нет |
| `macos-arm64` | macOS arm64 | `.zip` | нет |

Формат выбирается аргументом `--artifact`:

| Artifact | Где поддержан | Итоговый файл |
|---|---|---|
| `archive` | все поддержанные цели | `.tar.gz` для Linux, `.zip` для Windows/macOS |
| `appimage` | `linux-x86_64` | `.AppImage` |
| `onefile` | `windows-x86_64` | один `.exe` |

Если `--artifact` не указан, используется `archive`, то есть старое поведение.

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

Можно посмотреть план для другого формата или другой платформы:

```bash
uv run --extra build python build_tools/portable_build.py --platform windows-x86_64 --dry-run
uv run --extra build python build_tools/portable_build.py --artifact appimage --dry-run
uv run --extra build python build_tools/portable_build.py --platform windows-x86_64 --artifact onefile --dry-run
```

Фактическая сборка чужой платформы намеренно запрещена: wrapper не пытается
кросс-компилировать.

## Сборка текущей платформы

```bash
uv run --extra build python build_tools/portable_build.py --clean
```

Явно выбрать формат можно через `--artifact`:

```bash
uv run --extra build python build_tools/portable_build.py --artifact archive --clean
uv run --extra build python build_tools/portable_build.py --artifact appimage --clean
uv run --extra build python build_tools/portable_build.py --artifact onefile --clean
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
    magicborder-0.1.0-linux-x86_64.AppImage
  windows-x86_64/
    magicborder-0.1.0-windows-x86_64.zip
    magicborder-0.1.0-windows-x86_64.exe
  macos-x86_64/
    magicborder-0.1.0-macos-x86_64.zip
  macos-arm64/
    magicborder-0.1.0-macos-arm64.zip
```

Внутри archive-артефакта находится одна корневая папка:

```text
magicborder-0.1.0-<platform>/
```

В сборку включаются:

- исполняемый bundle PyInstaller;
- `oiv_ampelometric_scales.json`;
- `magicborder/assets`.

После сборки wrapper выполняет smoke-test артефакта:

- для `archive`: архив существует, не пустой, внутри есть исполняемый файл,
  OIV JSON и assets;
- для `onefile`: exe-файл существует, не пустой и имеет ожидаемое имя;
- для `appimage`: AppImage существует, не пустой и имеет executable bit.

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

Для AppImage используется тот же PyInstaller onedir bundle, затем wrapper
формирует AppDir и вызывает внешний `appimagetool`. Wrapper не скачивает
`appimagetool` автоматически. Указать инструмент можно одним из способов:

```bash
uv run --extra build python build_tools/portable_build.py --artifact appimage --appimagetool /path/to/appimagetool
APPIMAGETOOL=/path/to/appimagetool uv run --extra build python build_tools/portable_build.py --artifact appimage
```

Если `--appimagetool` и `APPIMAGETOOL` не указаны, wrapper ищет `appimagetool` в
`PATH`. AppImage может зависеть от glibc системы сборки и от поддержки FUSE на
целевой машине; запуск AppImage не является обязательной частью smoke-test.

#### Что такое appimagetool

`appimagetool` - это низкоуровневый Linux-инструмент проекта AppImage, который
создает итоговый `.AppImage` из уже подготовленного каталога `AppDir`. В нашем
wrapper-е `AppDir` собирается из PyInstaller onedir bundle: туда кладутся
`AppRun`, desktop-файл, иконка, исполняемый файл MagicBorder и runtime-данные.
После этого `appimagetool` упаковывает этот каталог в один исполняемый AppImage.

Официальная документация AppImage описывает `appimagetool` как самый простой
способ создать AppImage из существующего `AppDir`:

```text
https://docs.appimage.org/introduction/software-overview.html#appimagetool
```

Репозиторий и готовые сборки инструмента находятся здесь:

```text
https://github.com/AppImage/appimagetool
https://github.com/AppImage/appimagetool/releases/continuous
```

Важно: `appimagetool` сам распространяется как AppImage. Это не Python-пакет и
не зависимость приложения MagicBorder, поэтому его не нужно добавлять в
`pyproject.toml` или `uv.lock`.

#### Локальная установка appimagetool в проект

Рекомендуемый локальный вариант - скачать инструмент в приватный каталог проекта,
который не попадает в git:

```bash
mkdir -p .tools
curl -L \
  https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage \
  -o .tools/appimagetool-x86_64.AppImage
chmod +x .tools/appimagetool-x86_64.AppImage
```

Проверить, что файл запускается:

```bash
.tools/appimagetool-x86_64.AppImage --version
```

После этого можно собрать AppImage одним из двух способов:

```bash
uv run --extra build python build_tools/portable_build.py \
  --artifact appimage \
  --appimagetool "$PWD/.tools/appimagetool-x86_64.AppImage" \
  --clean
```

или через переменную окружения:

```bash
APPIMAGETOOL="$PWD/.tools/appimagetool-x86_64.AppImage" \
  uv run --extra build python build_tools/portable_build.py --artifact appimage --clean
```

Если система не умеет запускать AppImage через FUSE, можно использовать режим
extract-and-run. Wrapper передает окружение в `appimagetool`, поэтому достаточно
задать переменную перед сборкой:

```bash
APPIMAGE_EXTRACT_AND_RUN=1 \
APPIMAGETOOL="$PWD/.tools/appimagetool-x86_64.AppImage" \
  uv run --extra build python build_tools/portable_build.py --artifact appimage --clean
```

На Ubuntu 22.04 для старых AppImage часто нужен пакет `libfuse2`, а на
Ubuntu 24.04 и Debian 13 - `libfuse2t64`. Устанавливать системные FUSE-пакеты
нужно только если локальный запуск AppImage действительно падает с ошибкой FUSE.

Новый `appimagetool` может сам загрузить AppImage runtime во время создания
артефакта. Если сборка должна быть полностью офлайн, заранее подготовьте runtime
и используйте опцию самого `appimagetool` `--runtime-file`; в текущем wrapper-е
этот сценарий пока считается ручной расширенной настройкой.

### Windows

Windows-сборку нужно выполнять на Windows x86_64. В проекте для Windows
используется `PyQt5-Qt5==5.15.2`, потому что свежая ветка `PyQt5-Qt5 5.15.19`
не публикует Windows wheels.

Формат `--artifact onefile` включает PyInstaller `--onefile` и выдает один
распространяемый `.exe` без дополнительного `.zip`:

```powershell
uv run --extra build python build_tools/portable_build.py --artifact onefile --clean
```

При запуске такой exe может временно распаковывать внутренние файлы. Это
нормальное поведение PyInstaller onefile и не означает, что распространять нужно
несколько файлов.

### macOS

macOS-сборки нужно выполнять на macOS. Для x86_64 и arm64 предполагаются отдельные
артефакты. На этапе первого wrapper-а universal2-сборка не настраивается.

## Служебные файлы

Сборочный механизм состоит из:

```text
build_tools/
  portable_build.py
  assets/
    magicborder.svg
  backends/
    pyinstaller.py
  manifests/
    magicborder.toml
```

`build_tools/portable_build.py` отвечает за CLI, platform detection, упаковку
финального артефакта и smoke-test. `build_tools/backends/pyinstaller.py`
содержит только backend-специфичную команду PyInstaller. AppImage-иконка
`build_tools/assets/magicborder.svg` используется только при формировании AppDir.

Реальные build-output файлы игнорируются git. В репозитории оставлен только
`dist/.gitkeep`, чтобы каталог назначения существовал.
