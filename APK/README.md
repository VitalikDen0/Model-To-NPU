# Model-to-NPU Android Application

**Current target:** SDXL

> [!WARNING]
> The Android app and the surrounding scripts are currently being re-tested together with the rest of the repository. The app is usable, but the documentation is intentionally conservative until the next full validation pass is complete.

## Status: Working / In active validation

APK для генерации изображений непосредственно на телефоне через Qualcomm NPU.  
Текущая реализованная цель приложения — **SDXL Lightning**.  
**Полностью автономно** — ПК не нужен после деплоя моделей.

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
│   su --mount-master             │
│   → Termux Python3              │
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
- **Root**: Magisk (`su --mount-master` для mount namespace isolation)
- **Termux**: Python 3.13+, numpy, Pillow
- **QNN**: Context binaries предварительно задеплоены (см. `scripts/deploy_to_phone.py`)

## Installation

### 1. Подготовка Termux

```bash
pkg install python python-numpy python-pillow
```

### 2. Деплой моделей (с ПК)

```bash
python scripts/deploy_to_phone.py \
  --contexts-dir /path/to/contexts \
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
- **Seed** — воспроизводимая генерация (пусто = случайный)
- **Контрастирование** — percentile-based contrast stretch
- **Сохранение** — в галерею
- **Stop** — прерывание генерации
- **Прогресс-бар** — этапы CLIP -> UNet -> VAE
- **⚙️ Settings** — путь к моделям и Python

## Model settings

Через ⚙️ в ActionBar можно:
- изменить **путь к папке моделей** (по умолчанию `/data/local/tmp/sdxl_qnn`);
- изменить **путь к Python** (по умолчанию Termux Python3);
- нажать **Проверить файлы** — увидеть наличие нужных файлов на телефоне;
- сделать **Сброс** — вернуть значения по умолчанию.

## Performance

| Этап | Время |
|------|-------|
| CLIP-L + CLIP-G | ~2.3 с |
| UNet encoder + decoder × 8 шагов (без CFG) | ~113 с |
| UNet encoder + decoder × 8 шагов (CFG=3.5) | ~236 с |
| VAE | ~10 с |
| **Итого (без CFG)** | **~126 с** |
| **Итого (CFG=3.5)** | **~251 с** |

## Files on the phone

```text
/data/local/tmp/sdxl_qnn/       (или ваш путь из Настроек)
├── context/
│   ├── clip_l.serialized.bin.bin
│   ├── clip_g.serialized.bin.bin
│   ├── unet_encoder_fp16.serialized.bin.bin
│   ├── unet_decoder_fp16.serialized.bin.bin
│   └── vae_decoder.serialized.bin.bin
├── phone_gen/
│   ├── generate.py
│   └── tokenizer/
│       ├── vocab.json
│       └── merges.txt
├── lib/    (QNN runtime libs)
├── model/  (optional extra model libs in some flows)
├── bin/    (qnn-net-run)
└── outputs/
```

## Technical notes

- APK вызывает `phone_generate.py` через `su --mount-master` + Termux Python
- `--mount-master` обязателен из-за mount namespace isolation в Android 10+
- stdout парсится в реальном времени для отображения прогресса
- результат (PNG) загружается через `BitmapFactory.decodeFile()`
- сохранение в галерею идёт через `MediaStore` API
