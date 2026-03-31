# Model-to-NPU Android Application

**Языки:** [Русский](README.md) | [English](README_EN.md)

**Current target:** SDXL

> [!WARNING]
> The Android app and the surrounding scripts are currently being re-tested together with the rest of the repository. The app is usable, but the documentation is intentionally conservative until the next full validation pass is complete.

## Status: Working / In active validation

APK для генерации изображений непосредственно на телефоне через Qualcomm NPU.  
Текущая реализованная цель приложения — **SDXL Lightning**.  
**Полностью автономно** — ПК не нужен после деплоя моделей.

Текущая документированная версия APK: **`0.2.3`**.

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

Свежий `v0.2.3` reuse-path сдвинул README-visible marker до **78.0 s total**. На текущем fast path первые guided-шаги UNet идут примерно как **12.2 → 10.4 → 9.9 → 9.8 s**, а первые `CFG=1.0` шаги — примерно как **7.4 → 7.4 → 6.2 → 6.5 s**. Отдельно live preview через TAESD после переноса на rebuilt **QNN GPU** path теперь держится примерно около **1.0 s** на шаг вместо старых **5.5–6.0 s** на CPU.

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

- APK запускает `phone_generate.py` без `su`, через обычный shell и настраиваемую Python-команду
- По умолчанию используется общая папка `/sdcard/Download/sdxl_qnn`
- APK `v0.2.3` перед запуском runtime явно экспортирует `SDXL_QNN_USE_MMAP=1`, `SDXL_QNN_PERF_PROFILE=sustained_high_performance`, включает live-температуры, автоматически добавляет `SDXL_QNN_CONFIG_FILE`, если рядом лежат `htp_backend_extensions_lightning.json` и `lib/libQnnHtpNetRunExtensions.so`, уводит временные `WORK_DIR` / preview / output-файлы в app cache, чтобы меньше упираться в shared storage, и корректно парсит preview-тайминги формата `QNN GPU ...ms`
- Live Preview теперь в первую очередь использует rebuilt QNN TAESD preview assets (`taesd_decoder.serialized.bin.bin` и/или `model/libTAESDDecoder.so`) через GPU backend, держится примерно около **1.0 s** на шаг и откатывается на `phone_gen/taesd_decoder.onnx` на CPU через `onnxruntime` только как fallback
- CFG выше `1.0` заметно замедляет генерацию, потому что phone-side runtime всё ещё считает и cond-, и uncond-ветку; при split UNet это означает существенно больше encoder/decoder-работы на каждый шаг даже после batching-оптимизаций
- Режим **½-CFG** прокидывает в phone runtime флаг `--prog-cfg` и держит guidance только на первых `ceil(steps / 2)` шагах; это компромиссный режим между скоростью и силой guidance
- status/output парсер теперь отдельно отображает live-строку `CPU / GPU / NPU`, не ломая основной прогресс
- stdout парсится в реальном времени для отображения прогресса
- результат (PNG) загружается через `BitmapFactory.decodeFile()`
- сохранение в галерею идёт через `MediaStore` API
