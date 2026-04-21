# Model-to-NPU Android Application

**Языки:** [Русский](README.md) | [English](README_EN.md)

**Current target:** SDXL

> [!WARNING]
> The Android app and the surrounding scripts are currently being re-tested together with the rest of the repository. The app is usable, but the documentation is intentionally conservative until the next full validation pass is complete.

## Status: Working / In active validation

APK для генерации изображений непосредственно на телефоне через Qualcomm NPU.  
Текущая реализованная цель приложения — **SDXL Lightning**.  
**Полностью автономно** — ПК не нужен после деплоя моделей.

Текущая документированная версия APK: **`0.4.4`**.

Текущая линия `0.4.4` уводит APK в более щадящий пользовательский режим: на главном экране оставлен только preset picker, ручной `WxH` скрыт, preview/final PNG для показа декодируются в экранный размер, а APK-side runtime теперь по умолчанию просит `sustained_high_performance` вместо `burst`, чтобы уменьшить шанс заметного лага телефона и краша приложения во время генерации.

Последний валидированный локальный review-замер на OnePlus 13 для стандартного пути (`seed=777`, `8` шагов, `CFG=3.5`, `--prog-cfg`, Live Preview OFF) дал **75.6 s total** на связке `burst` + native runtime accel; повтор с `basic` profiling остался практически тем же — **75.2 s total**. Для новой линии `0.4.4` в этой сессии пока подтверждена сборка APK, но свежий phone-side timing run ещё не записан.

Историческая заметка: старый лучший runtime-результат **62.0 s** относился к доресторовому состоянию телефона. Сам тест был реальным, но после factory reset точное phone-side состояние, скриншоты и вспомогательные технические артефакты не были сохранены, поэтому репозиторий больше не может честно воспроизвести или документально доказать именно эту цепочку как текущую.

Важно: производительность в этой связке в первую очередь определяется `phone_generate.py` (на телефоне: `phone_gen/generate.py`), поэтому одна и та же версия APK может работать быстрее после обновления только runtime-скрипта.

## Architecture

```text
┌─────────────────────────────────┐
│         MainActivity            │
│  ┌───────────────────────────┐  │
│  │  Промпт + Негативный      │  │
│  │  Seed / Steps / CFG       │  │
│  ├───────────────────────────┤  │
│  │     Image Preview         │  │
│  ├───────────────────────────┤  │
│  │  Generate / Stop / Save   │  │
│  └───────────────────────────┘  │
│         ⚙️ SettingsActivity     │
│  ┌───────────────────────────┐  │
│  │  Путь к моделям          │  │
│  │  Путь к Python           │  │
│  │  Проверка / Сброс        │  │
│  └───────────────────────────┘  │
├─────────────────────────────────┤
│   Shared Downloads path         │
│   → configurable Python3        │
│   → phone_generate.py           │
│   (BPE tokenizer + scheduler)   │
├─────────────────────────────────┤
│       QNN Runtime (HTP/NPU)     │
│  ┌───────┬───────┬───────────┐  │
│  │CLIP-L │CLIP-G │Split UNet │  │
│  │ FP16  │ FP16  │  FP16     │  │
│  ├───────┴───────┤enc + dec  │  │
│  │    VAE FP16   │           │  │
│  └───────────────┴───────────┘  │
└─────────────────────────────────┘
```

## Requirements

- **Телефон**: Snapdragon 8 Elite (SM8750) или аналог с Qualcomm NPU
- **Root**: не требуется для текущего пути по умолчанию
- **Termux**: Python 3.13+, numpy, Pillow, `termux-setup-storage`
- **TAESD live preview (optional)**: предпочитаемый путь — задеплоенные QNN TAESD assets на GPU; fallback — `onnxruntime` в Termux + `phone_gen/taesd_decoder.onnx`
- **Android 11+**: для чтения общей папки Downloads может понадобиться доступ ко всем файлам
- **QNN**: Context binaries предварительно задеплоены (см. `scripts/deploy_to_phone.py`)

## Installation

### 1. Подготовка Termux

```bash
pkg install python python-numpy python-pillow
python -m pip install onnxruntime   # optional, only for Live Preview
termux-setup-storage
```

### 2. Деплой моделей (с ПК)

```bash
python scripts/deploy_to_phone.py \
  --contexts-dir /path/to/contexts \
  --phone-base /sdcard/Download/sdxl_qnn \
  --qnn-lib-dir /path/to/qnn/lib \
  --qnn-bin-dir /path/to/qnn/bin
```

### 3. Сборка и установка APK

```bash
cd APK
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Features

- **Промпт и негативный промпт** — полная BPE-токенизация CLIP
- **Steps** (1–20) — число шагов UNet denoising
- **CFG** (1.0–7.0) — classifier-free guidance
- **½-CFG** — guidance только на первых `ceil(steps / 2)` шагах, чтобы не платить полный CFG-штраф до самого конца
- **Seed** — воспроизводимая генерация (пусто = случайный)
- **Контрастирование** — percentile-based contrast stretch
- **Live Preview (TAESD)** — промежуточный preview, который теперь предпочитает QNN GPU путь и откатывается на CPU-side ONNX только как fallback
- **Bundled offline runtime (optional)** — если в APK упакованы offline Termux assets, приложение умеет их распаковать во внутреннее хранилище и использовать как локальный Python runtime
- **Сохранение** — в галерею
- **Stop** — прерывание генерации
- **Прогресс-бар** — этапы CLIP -> UNet -> VAE
- **Live temperatures** — строка CPU / GPU / NPU прямо во время генерации
- **⚙️ Settings** — путь к моделям и Python

## Model settings

Через ⚙️ в ActionBar можно:

- изменить **путь к папке моделей** (по умолчанию `/sdcard/Download/sdxl_qnn`);
- изменить **команду или путь к Python** (по умолчанию `python3`);
- нажать **Проверить файлы** — увидеть наличие нужных файлов на телефоне;
- сделать **Сброс** — вернуть значения по умолчанию.

## Performance

| Этап | Время |
| ---- | ------ |
| CLIP-L + CLIP-G | ~2.3 с |
| UNet encoder + decoder × 8 шагов (без CFG) | ~113 с |
| UNet encoder + decoder × 8 шагов (CFG=3.5) | ~236 с |
| VAE | ~10 с |
| **Итого (без CFG)** | **~126 с** |
| **Итого (CFG=3.5)** | **~251 с** |

Контрольный прогон `v0.1.3` из этой сессии (`1024×1024`, `8` шагов, `CFG=1.0`, `mmap` ON) дал **104.4 s total**: `CLIP 1.993 s`, `UNet 91.466 s`, `VAE 8.992 s`.

Лучший session-validated путь для `v0.2.0` с `sustained_high_performance`, backend extensions и `--prog-cfg` (`8` шагов, `CFG=3.5`) дал **79.7–80.6 s total** на OnePlus 13. При этом в полном прогоне live-лог температуры обычно держался около **CPU ~59–70°C**, **GPU ~50–52°C**, **NPU ~57–72°C**, с короткими NPU-пиками примерно до **78°C**.

Для текущего состояния APK/runtime полезно держать в голове три маркера:

- APK screenshot marker (Live Preview ON): около **78.0 s total**;
- текущий rebuilt-phone local review (Live Preview OFF, `burst` + native runtime accel): **75.6 s total** при `seed=777`, `steps=8`, `CFG=3.5`, `--prog-cfg`, со стадиями `CLIP 2.774 s`, `UNet 66.639 s`, `VAE 2.960 s`;
- историческая до-reset заметка: **62.0 s total** на старом пути `v0.2.3` со стадиями `CLIP 1.787 s`, `UNet 55.980 s`, `VAE 3.138 s`.

Прогрессия шагов UNet в историческом прогоне `62.0 s`:

- CFG шаги 1..4: **9.765 → 8.230 → 8.386 → 7.936 s**;
- no-guidance шаги 5..8: **5.377 → 5.513 → 5.294 → 5.479 s**.

Live preview через TAESD после переноса на rebuilt **QNN GPU** path держится примерно около **1.0 s** на шаг вместо старых **5.5–6.0 s** на CPU; этот preview/UI-overhead и объясняет разницу между APK-скриншотами и no-preview runtime-замерами.

Практический смысл такой: в текущем пересобранном состоянии телефона нормой остаётся класс **~75–78 s**, а маркер **62.0 s** нужно читать только как исторический до-reset runtime note, а не как гарантированно воспроизводимую текущую цифру.

## Files on the phone

```text
/sdcard/Download/sdxl_qnn/      (или ваш путь из Настроек)
├── context/
│   ├── clip_l.serialized.bin.bin
│   ├── clip_g.serialized.bin.bin
│   ├── unet_encoder_fp16.serialized.bin.bin
│   ├── unet_decoder_fp16.serialized.bin.bin
│   └── vae_decoder.serialized.bin.bin
├── phone_gen/
│   ├── generate.py
│   ├── taesd_decoder.onnx
│   └── tokenizer/
│       ├── vocab.json
│       └── merges.txt
├── htp_backend_extensions_lightning.json
├── htp_backend_ext_config_lightning.json
├── lib/    (QNN runtime libs)
│   └── libQnnHtpNetRunExtensions.so   (optional, enables backend extensions)
├── model/  (optional extra model libs in some flows)
├── bin/    (qnn-net-run)
└── outputs/
```

## Technical notes

- Версия APK и скорость runtime не всегда меняются синхронно: ускорения часто приходят из обновлённого `phone_generate.py`, даже если номер APK остаётся тем же.
- APK запускает `phone_generate.py` без `su`, через обычный shell и настраиваемую Python-команду
- По умолчанию используется общая папка `/sdcard/Download/sdxl_qnn`
- APK `v0.4.4` скрывает ручной `WxH` на главном экране и всегда приводит генерацию к ближайшему валидированному preset'у, чтобы пользовательский путь не уходил в произвольные размеры с более высоким риском лагов/нестабильности
- APK `v0.4.4` декодирует preview/final PNG для показа через screen-sized `ImageDecoder` path (с `BitmapFactory` fallback), чтобы не держать в UI лишние full-resolution bitmap'ы
- APK `v0.4.4` экспортирует `SDXL_QNN_PERF_PROFILE=sustained_high_performance` именно из Android-приложения, уменьшая пиковую агрессивность runtime относительно старого APK-side `burst` default
- Обновлённый публичный asset `v0.4.3` чинит реальную regression path из `v0.4.3`: если приложение экспортирует bundled QNN runtime paths, phone runtime больше не должен молча перескакивать на устаревший `/data/local/tmp/sdxl_qnn`
- Обновлённый публичный asset `v0.4.3` теперь действительно включает в payload `qnn-net-run` и базовый набор QNN HTP/System библиотек, так что bundled fast path меньше зависит от случайно оставшихся phone-side runtime файлов
- Обновлённый публичный asset `v0.4.3` корректно перестейдживает bundled backend-extension config с относительными путями и тем самым не теряет backend extensions только из-за app-private extraction path
- Обновлённый публичный asset `v0.4.3` агрессивнее убирает старые preview/final bitmap'ы и отменяет stale preview poller callbacks, чтобы не держать лишние `Bitmap` и delayed callbacks дольше нужного
- Обновлённый публичный asset `v0.4.3` уводит decode `preview_current.png` с UI thread на отдельный executor и при live preview включает `SDXL_QNN_PREVIEW_STRIDE=4`, чтобы preview заметно меньше мешал UI во время генерации
- Обновлённый публичный asset `v0.4.3` убирает `832x480` из общего SDXL size picker и оставляет его только в WAN-режиме
- APK `v0.4.3` переводит prewarm c приватного stdin/stdout child-сервера на shared FIFO-backed `qnn-multi-context-server` с детерминированными ID контекстов, поэтому прогрев при открытии приложения теперь может реально переиспользоваться последующим foreground generate-запуском
- APK `v0.4.3` заставляет prewarm и foreground generate использовать один и тот же app-cache `SDXL_QNN_WORK_DIR` и включает `SDXL_QNN_SHARED_SERVER=1`, что делает multi-resolution context reuse практическим, а не декоративным
- APK `v0.4.3` теперь принудительно переэкстрагирует bundled runtime payload, когда его версия в APK изменилась, так что новые `generate.py` / server-бинарники действительно заменяют старые on-device копии
- APK `v0.4.3` использует bundled runtime с исправленным ожиданием shared FIFO IPC readiness: QNN server после `READY` дополнительно ждётся до реального появления request/response FIFO, чтобы первый `LOAD` не падал на старте из-за гонки
- Phone runtime `v0.4.3` теперь ретраит transient `BlockingIOError` / `InterruptedError` при чтении shared response FIFO, чтобы prewarm не умирал на ранней non-blocking IPC-гонке сразу после `READY`
- APK `v0.4.3` больше не должен плодить duplicate prewarm helper'ы: app-open launch защищён `prewarmStartQueued`, warm shared-server window учитывается перед повторным прогревом, а Python запускается через `exec`, чтобы приложение отслеживало реальный helper-процесс вместо промежуточной shell-обёртки
- Обновлённый публичный asset `v0.4.3` теперь выгружает shared prewarm через 30 секунд простоя не только после сворачивания приложения, но и после app-open prewarm / завершённой генерации в foreground, чтобы не держать QNN runtime в памяти бесконечно
- Обновлённый публичный asset `v0.4.3` при наличии файлов в payload экспортирует bundled TAESD ONNX/QNN preview-пути (`taesd_decoder.onnx`, optional `taesd_decoder.serialized.bin.bin`, `libTAESDDecoder.so`, `libQnnGpu.so`) и тем самым заметно меньше зависит от старых preview-артефактов в shared storage
- APK `v0.4.1` перед запуском runtime явно экспортирует `SDXL_QNN_USE_MMAP=1`, `SDXL_QNN_PERF_PROFILE=burst`, жёстко отключает daemon fast-path (`SDXL_QNN_USE_DAEMON=0`), включает async/prestage/prewarm-флаги для текущего Python runtime, включает live-температуры, автоматически добавляет `SDXL_QNN_CONFIG_FILE`, если рядом лежат `htp_backend_extensions_lightning.json` и `lib/libQnnHtpNetRunExtensions.so`, уводит временные `WORK_DIR` / preview / output-файлы в app cache, отключает дополнительное PNG-сжатие финального изображения ради небольшого снижения save-overhead и корректно парсит preview-тайминги формата `QNN GPU ...ms`
- APK `v0.4.1` сначала автоматически извлекает bundled offline runtime, потом ищет рабочий Python, а если обычный app shell не видит Termux-private `python3`, автоматически переключается на root shell вместо немого `127`
- APK `v0.4.1` теперь также тащит внутри bundled phone runtime payload и запускает актуальный `generate.py` из assets, так что устаревший `/sdcard/Download/sdxl_qnn/phone_gen/generate.py` больше не ломает новые `Size` / `--width` / `--height` параметры
- если в bundled payload есть `qnn-multi-context-server`, `qnn-context-runner` и/или `libsdxl_runtime_accel.so`, APK `v0.4.1` предпочитает именно эти артефакты и тем самым держит актуальный fast-path ближе к APK, а не к случайно устаревшим файлам в shared storage
- `scripts/deploy_to_phone.py` для `v0.4.1` умеет подхватывать `libQnnHtpV79Skel.so` не только из `aarch64-android`, но и из sibling-папки `hexagon-v79/unsigned`, а также деплоить `phone_runtime_accel.py`, `qnn-multi-context-server`, optional `libsdxl_runtime_accel.so` и рекурсивно раскладывать resolution-scoped context directories
- Live Preview теперь в первую очередь использует rebuilt QNN TAESD preview assets (`taesd_decoder.serialized.bin.bin` и/или `model/libTAESDDecoder.so`) через GPU backend, держится примерно около **1.0 s** на шаг и откатывается на `phone_gen/taesd_decoder.onnx` на CPU через `onnxruntime` только как fallback
- CFG выше `1.0` заметно замедляет генерацию, потому что phone-side runtime всё ещё считает и cond-, и uncond-ветку; при split UNet это означает существенно больше encoder/decoder-работы на каждый шаг даже после batching-оптимизаций
- Режим **½-CFG** прокидывает в phone runtime флаг `--prog-cfg` и держит guidance только на первых `ceil(steps / 2)` шагах; это компромиссный режим между скоростью и силой guidance
- status/output парсер теперь отдельно отображает live-строку `CPU / GPU / NPU`, не ломая основной прогресс
- stdout парсится в реальном времени для отображения прогресса
- результат (PNG) для показа теперь декодируется в screen-sized bitmap через `ImageDecoder` с `BitmapFactory` fallback
- сохранение в галерею идёт через `MediaStore` API
