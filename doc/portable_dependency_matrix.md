# Матрица зависимостей для portable-сборок

Дата аудита: 2026-06-01.

Документ фиксирует зависимостную базу MagicBorder для следующего этапа, где будет
настраиваться сборка переносимых дистрибутивов. Сами сборочные механизмы здесь не
вводятся: не добавлены PyInstaller/Nuitka/cx_Freeze, не создан каталог `dist/` и
не собирались платформенные архивы.

## Выбор Python

Для portable-сборок выбран диапазон CPython `>=3.11,<3.14`. Рекомендуемая версия
для первых сборочных сценариев - CPython 3.12.x.

Причины:
- Python 3.11-3.13 покрывается wheels для `numpy`, `opencv-python-headless`,
  `Pillow`, `PyQt5`, `PyQt5-Qt5` и `PyQt5-sip` на основных x86_64-платформах.
- Python 3.10 приводил к отдельной ветке `numpy` в `uv.lock`, что ухудшало
  единообразие portable-сборок.
- Python 3.14 пока не включен в проектный диапазон: часть wheels уже есть, но
  для упаковщиков portable-приложений это более рискованная цель на текущем
  этапе.

## Итоговые ограничения

В `pyproject.toml` зафиксированы диапазоны с верхними границами для бинарных
пакетов:

| Пакет | Назначение | Ограничение в проекте | Версия в `uv.lock` |
|---|---|---|---|
| `numpy` | численные вычисления и массивы изображений | `>=2.4,<2.5` | `2.4.6` |
| `opencv-python-headless` | обработка изображений без GUI-зависимостей OpenCV | `>=4.13.0.92,<4.14` | `4.13.0.92` |
| `Pillow` | чтение и запись растровых изображений | `>=12.2,<13` | `12.2.0` |
| `PyQt5` | Python bindings для Qt 5 UI | `>=5.15.11,<5.16` | `5.15.11` |
| `PyQt5-Qt5` | Qt runtime для PyQt5 | `==5.15.2` на Windows, `>=5.15.19,<5.16` на остальных ОС | `5.15.2` / `5.15.19` |
| `PyQt5-sip` | runtime-модуль PyQt5 | `>=12.18,<13` | `12.18.0` |
| `hatchling` | backend сборки wheel | `>=1.29,<2` | не runtime-зависимость |

## Wheel-матрица

Статусы ниже основаны на PyPI release files для выбранных версий пакетов.

| Пакет | Linux x86_64 | Windows x86_64 | macOS x86_64 | macOS arm64 | Linux aarch64 | Комментарий |
|---|---|---|---|---|---|---|
| `PyQt5 5.15.11` | Да, `manylinux_2_17_x86_64` | Да, `win_amd64` | Да, `macosx_11_0_x86_64` | Да, `macosx_11_0_arm64` | Нет wheel | Основной UI-пакет. Linux aarch64 потребует отдельного решения. |
| `PyQt5-Qt5 5.15.19` | Да, `manylinux2014_x86_64` | Нет | Да, `macosx_10_13_x86_64` | Да, `macosx_11_0_arm64` | Нет wheel | Используется для Linux/macOS. |
| `PyQt5-Qt5 5.15.2` | Да, `manylinux2014_x86_64` | Да, `win_amd64` | Нет | Нет | Нет wheel | Используется только на Windows, потому что свежий `PyQt5-Qt5 5.15.19` не публикует Windows wheels. |
| `PyQt5-sip 12.18.0` | Да, `manylinux*_x86_64` | Да, `win_amd64` | Да, `macosx_*_universal2` | Да, `macosx_*_universal2` | Нет wheel | Совместим с Python 3.11-3.13. |
| `numpy 2.4.6` | Да, `manylinux_2_27_x86_64` | Да, `win_amd64` | Да | Да | Да, `manylinux_2_27_aarch64` | Покрывает Python 3.11-3.13 одной веткой. |
| `opencv-python-headless 4.13.0.92` | Да, `manylinux2014_x86_64` и `manylinux_2_28_x86_64` | Да, `win_amd64` | Да, `macosx_14_0_x86_64` | Да, `macosx_13_0_arm64` | Да | Для macOS x86_64 wheel рассчитан на новую macOS; это нужно учесть при выборе минимальной версии ОС. |
| `Pillow 12.2.0` | Да | Да, `win_amd64` | Да | Да | Да | Широкое покрытие wheels для Python 3.11-3.13. |

## Платформенные выводы

Минимально готовые цели для следующего этапа:
- Linux x86_64;
- Windows x86_64;
- macOS x86_64, с учетом ограничения OpenCV wheel по версии macOS;
- macOS arm64.

Пока не считать готовой portable-целью:
- Linux aarch64. У `numpy`, `opencv-python-headless` и `Pillow` wheels есть, но
  Qt-стек `PyQt5`/`PyQt5-Qt5` не публикует Linux aarch64 wheels для выбранной
  связки. Для этой платформы потребуется отдельная стратегия: другой Qt binding,
  собственная сборка Qt/PyQt или отказ от готового wheel-only сценария.

## Qt-связка

Проект остается на `PyQt5`; переход на `PyQt6` или другой UI-фреймворк в рамках
этой подготовки не выполнялся.

Выбранная связка:
- `PyQt5 >=5.15.11,<5.16`;
- `PyQt5-sip >=12.18,<13`;
- `PyQt5-Qt5 ==5.15.2` для Windows;
- `PyQt5-Qt5 >=5.15.19,<5.16` для Linux/macOS.

Отдельный Windows marker нужен из-за публикации wheels `PyQt5-Qt5`: версия
`5.15.19` покрывает Linux/macOS, но не Windows x86_64, а `5.15.2` публикует
Windows wheel.

## Проверки

Выполнено:
- `uv tree` до изменения зависимостей:
  - проект резолвился с `numpy 2.4.4`, `opencv-python-headless 4.13.0.92`,
    `Pillow 12.2.0`, `PyQt5 5.15.11`, `PyQt5-Qt5 5.15.18`, `PyQt5-sip 12.18.0`.
- `uv lock --upgrade` после изменения зависимостей:
  - lock обновлен до `numpy 2.4.6`;
  - не-Windows `PyQt5-Qt5` обновлен до `5.15.19`;
  - Python marker в lock сокращен до `sys_platform == 'win32'` и
    `sys_platform != 'win32'`.
- `uv lock --check`:
  - lock согласован с `pyproject.toml`.
- Короткая проверка импортов:
  - `PyQt5` импортируется;
  - `cv2 4.13.0`;
  - `numpy 2.4.6`;
  - `Pillow 12.2.0`.
- `uv run pytest`:
  - `114 passed`.

## Источники

- PyPI JSON API, `PyQt5`: <https://pypi.org/pypi/PyQt5/json>
- PyPI JSON API, `PyQt5-Qt5`: <https://pypi.org/pypi/PyQt5-Qt5/json>
- PyPI JSON API, `PyQt5-sip`: <https://pypi.org/pypi/PyQt5-sip/json>
- PyPI JSON API, `numpy`: <https://pypi.org/pypi/numpy/json>
- PyPI JSON API, `opencv-python-headless`: <https://pypi.org/pypi/opencv-python-headless/json>
- PyPI JSON API, `Pillow`: <https://pypi.org/pypi/Pillow/json>
- Локальный `uv.lock`, обновленный командой `uv lock --upgrade` 2026-06-01.
