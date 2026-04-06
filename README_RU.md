# Model-to-NPU Pipeline for Snapdragon

**Языки:** [English](README_EN.md) | [Русский](README_RU.md)

> [!WARNING]
> Репозиторий сейчас проходит повторную end-to-end проверку.
> Структура и команды ниже отражают последнее известное рабочее состояние, но полный прогон «с чистого ПК до финальной генерации на телефоне» ещё перепроверяется.

<p align="center">
  <b>Репозиторий для model-to-NPU pipeline'ов на Qualcomm Snapdragon</b><br>
  Текущий реализованный pipeline: <b>SDXL на Qualcomm Hexagon NPU</b>.
</p>

---

## Что это?

Этот репозиторий задуман как общее место для нескольких **model-specific pipeline'ов**, нацеленных на Qualcomm Snapdragon NPU.

- для каждой семьи моделей предполагается своя папка;
- **текущая реализованная папка** — `SDXL/`;
- **текущее Android-приложение** лежит в `APK/`;
- общие deploy-скрипты и вспомогательные файлы живут в `scripts/`, `tokenizer/` и в корне.

Сейчас реально реализованный и задокументированный путь — это **Stable Diffusion XL**, работающий **нативно на NPU телефона** (Hexagon HTP). Текущий SDXL pipeline использует CLIP-L, CLIP-G, Split UNet (encoder + decoder) и VAE прямо на устройстве.

**Текущее протестированное сочетание:** [WAI Illustrious SDXL v1.60](https://civitai.com/models/827184/wai-illustrious-sdxl?modelVersionId=2514310) + [SDXL-Lightning 8-step LoRA](https://huggingface.co/ByteDance/SDXL-Lightning) (ByteDance)

> **Важно по скорости:** публичные тайминги, APK-скриншоты и примеры изображений в этом репозитории предполагают, что **Lightning LoRA уже замержена в UNet**. Это не просто косметический шаг: именно так устроен практический быстрый путь здесь. Если запускать базовую ветку SDXL без merge Lightning LoRA, на телефоне всё получается заметно медленнее, и сравнивать такие прогоны с цифрами ниже уже некорректно.
> **Важно по разрешению:** текущие экспорты, context binaries, preview и примеры в документации рассчитаны именно на **1024×1024**.
> **Важно:** структура репозитория уже делается шире SDXL, но фактически проверенный pipeline здесь пока SDXL-first.

## Текущий статус

- **Направление репозитория:** multi-model pipeline'ы под Snapdragon NPU
- **Сейчас реализованная семья:** `SDXL/`
- **Текущая цель APK:** SDXL
- **Статус скриптов:** повторная проверка полной воспроизводимости
- **Статус документации:** обновлена под текущую известную структуру

Для живого примера того, как сейчас выглядит SDXL-папка на телефоне после деплоя, см. [`examples/phone-sdxl-qnn-layout.md`](examples/phone-sdxl-qnn-layout.md).
Также добавлен небольшой rooted-набор артефактов в [`examples/rooted-phone-sample/`](examples/rooted-phone-sample/) — как справочный и учебный пример.

Для практических подводных камней и накопленных технических заметок см. [`SDXL/LESSONS_LEARNED.md`](SDXL/LESSONS_LEARNED.md) и русскую версию [`SDXL/LESSONS_LEARNED_RU.md`](SDXL/LESSONS_LEARNED_RU.md).

Для отдельной ревизии текущей структуры SDXL UNet, split-границ и зон риска при квантовании см. [`SDXL/UNET_QUANTIZATION_REVIEW.md`](SDXL/UNET_QUANTIZATION_REVIEW.md) и [`SDXL/UNET_QUANTIZATION_REVIEW_RU.md`](SDXL/UNET_QUANTIZATION_REVIEW_RU.md).

Для свежего разбора runtime-overhead, эффекта `mmap` и контрольных цифр после `0.1.3` см. [`SDXL/UNET_OVERHEAD_REVIEW.md`](SDXL/UNET_OVERHEAD_REVIEW.md) и [`SDXL/UNET_OVERHEAD_REVIEW_RU.md`](SDXL/UNET_OVERHEAD_REVIEW_RU.md).

Для карты всех текущих скриптов в `SDXL/` см. [`SDXL/SCRIPTS_OVERVIEW.md`](SDXL/SCRIPTS_OVERVIEW.md) и [`SDXL/SCRIPTS_OVERVIEW_RU.md`](SDXL/SCRIPTS_OVERVIEW_RU.md).

## Требования для текущего SDXL pipeline

### Телефон

| Компонент | Требование |
| --------- | ---------- |
| **SoC** | Qualcomm Snapdragon 8 Elite (SM8750) или совместимый с QNN HTP |
| **RAM** | 16 GB (пик ~12 GB, свободно >= 6 GB) |
| **Хранилище** | ~10 GB для моделей и context binary в общей папке вроде `/sdcard/Download/sdxl_qnn` |
| **Root** | Не требуется для текущей layout-схемы по умолчанию |
| **Termux** | Python 3.13+, numpy, Pillow, `termux-setup-storage` |

### ПК (для сборки текущего pipeline)

| Компонент | Версия |
| --------- | ------ |
| Python | 3.10.x (именно 3.10, не 3.11+) |
| QAIRT SDK | 2.31+ (Qualcomm AI Engine Direct) |
| Android NDK | r26+ (для сборки `.so`) |
| PyTorch | 2.x |
| Windows | 10/11 |

## Производительность

Замерено на OnePlus 13 (Snapdragon 8 Elite, 16 GB RAM):

| Этап | Время | Формат |
| ---- | ----- | ------ |
| CLIP-L | ~375 мс | FP16 |
| CLIP-G | ~1500 мс | FP16 |
| UNet encoder (1 шаг) | ~7.1 с | FP16 |
| UNet decoder (1 шаг) | ~7.1 с | FP16 |
| UNet (8 шагов, без CFG) | ~113 с | FP16 |
| UNet (8 шагов, CFG=3.5) | ~236 с | FP16 |
| VAE decoder | ~10 с | FP16 |
| **Итого (без CFG)** | **~126 с** | |
| **Итого (CFG=3.5)** | **~251 с** | |

Пик RAM: **~12 GB** из 16 GB  
Разрешение: **1024×1024** (фиксированное)

Свежий контрольный прогон `v0.1.3` с дефолтным `mmap` на OnePlus 13 (`1024×1024`, `8` шагов, `CFG=1.0`) дал **104.4 s total** (`CLIP 1.993 s`, `UNet 91.466 s`, `VAE 8.992 s`), то есть примерно на **17.1% быстрее** прежнего публичного no-CFG ориентира.

Новые настроенные прогоны `v0.2.0` с **живым логированием температур**, дефолтным `sustained_high_performance`, задеплоенными HTP backend extensions и **progressive CFG** (`8` шагов, `CFG=3.5`, `--prog-cfg`) дали **79.7–80.6 s total** на том же OnePlus 13:

- прогон 1: `CLIP 2.858 s`, `UNet 73.031 s`, `VAE 3.547 s`, **80.6 s total**;
- прогон 2: `CLIP 2.917 s`, `UNet 72.391 s`, `VAE 3.395 s`, **79.7 s total**.

Свежие прогоны `v0.2.3` после reuse-тюнинга сдвинули текущий README-visible marker до **78.0 s total** и убрали неприятное ощущение «почти плоской ~12-секундной полки» на первых guided-шагах UNet:

- первые четыре guided-шага на быстром CFG-пути сейчас идут примерно как **12.2 → 10.4 → 9.9 → 9.8 s**;
- первые четыре шага в `CFG=1.0` no-guidance пути — примерно **7.4 → 7.4 → 6.2 → 6.5 s** (с обычной погрешностью между прогонами);
- live preview через TAESD теперь в первую очередь использует rebuilt **QNN GPU** assets и держится примерно около **1.0 s/шаг**, вместо старого CPU-side ONNX preview пути на **5.5–6.0 s**.

В этих полных прогретых прогонах практическая термокартина держалась примерно в диапазоне **CPU ~59–70°C**, **GPU ~50–52°C**, **NPU ~57–72°C**, при кратковременных пиках NPU примерно до **78°C**. Самая первая строка с `CPU=88.8°C` выглядела как краткий скачок сенсора до стабилизации, а не как устойчивое состояние во время генерации.

## Быстрый старт

### 1. Подготовка окружения (ПК)

```bash
# Установить зависимости Python 3.10
pip install torch diffusers transformers safetensors onnx onnxruntime Pillow numpy

# Скачать QAIRT SDK
python scripts/download_qualcomm_sdk.py

# Скачать ADB (если ещё нет)
python scripts/download_adb.py
```

### 2. Сборка pipeline

> **Примечание:** `scripts/build_all.py` сейчас автоматизирует только ранние воспроизводимые шаги и специально **не** запускает вслепую все поздние QNN/deploy-стадии, пока они перепроверяются.
> **Важно про набор скриптов:** не все файлы в `SDXL/` нужны для кратчайшего happy-path. Большинство лабораторных/диагностических скриптов теперь сгруппированы в `SDXL/debug/`, а в корне `SDXL/` оставлен практический публичный поток.

```bash
# Экспериментальный помощник для ранних SDXL-этапов
python scripts/build_all.py --checkpoint path/to/model.safetensors
```

Также добавлен аккуратный beta-wrapper для текущего документированного пути:

```powershell
pwsh SDXL/run_end_to_end.ps1 -Checkpoint path/to/model.safetensors -ContextsDir path/to/context_binaries
```

Он специально разделяет воспроизводимые ранние шаги сборки и ещё beta/runtime/deploy-часть.

Или пошагово:

```bash
# 1. Скачать модель WAI Illustrious SDXL v1.60 и SDXL-Lightning LoRA

# 2. Конвертировать checkpoint в diffusers формат
python SDXL/convert_sdxl_checkpoint_to_diffusers.py

# 3. Замержить Lightning LoRA в UNet
python SDXL/bake_lora_into_unet.py

# 4. Экспортировать все компоненты в ONNX
python SDXL/export_clip_vae_to_onnx.py
python SDXL/export_sdxl_to_onnx.py

# 5. Конвертировать в QNN
python SDXL/convert_clip_vae_to_qnn.py
python SDXL/convert_lightning_to_qnn.py

# 6. Собрать Android model libraries (.so)
python SDXL/build_android_model_lib_windows.py
```

Дополнительный путь для live preview в APK / phone runtime:

```bash
# Экспорт tiny TAESD XL preview decoder
python SDXL/export_taesd_to_onnx.py --validate

# Деплой одного ONNX-файла в phone runtime
adb push D:/platform-tools/sdxl_npu/taesd_decoder/taesd_decoder.onnx /sdcard/Download/sdxl_qnn/phone_gen/

# Опционально: собрать и задеплоить TAESD QNN preview assets (предпочтительный путь)
python SDXL/convert_taesd_to_qnn.py --backend gpu

# Опционально только как fallback (если нужен CPU ONNX preview в Termux / APK)
python -m pip install onnxruntime
```

### 3. Деплой на телефон

```bash
python scripts/deploy_to_phone.py \
  --contexts-dir /path/to/context_binaries \
  --phone-base /sdcard/Download/sdxl_qnn \
  --qnn-lib-dir /path/to/qnn_sdk/lib/aarch64-android \
  --qnn-bin-dir /path/to/qnn_sdk/bin/aarch64-android
```

Если в `--qnn-lib-dir` есть `libQnnHtpNetRunExtensions.so`, deploy-скрипт теперь копирует и её тоже, так что phone runtime и APK смогут автоматически включить уже лежащий в репозитории путь через `htp_backend_extensions_lightning.json`. Тот же deploy helper также пытается докинуть optional TAESD preview assets (`taesd_decoder.serialized.bin.bin`, `libTAESDDecoder.so`, `libQnnGpu.so`, `qnn-gpu-target-server`), если они найдены локально.

### 4. Подготовка Termux (на телефоне)

```bash
pkg install python python-numpy python-pillow
python -m pip install onnxruntime   # опциональный CPU fallback для TAESD live preview
termux-setup-storage
```

### 5. Генерация

#### Standalone (в Termux на телефоне)

```bash
export PATH=/data/data/com.termux/files/usr/bin:$PATH
export SDXL_QNN_BASE=/sdcard/Download/sdxl_qnn
python3 "$SDXL_QNN_BASE/phone_gen/generate.py" "1girl, anime, cherry blossoms"
python3 "$SDXL_QNN_BASE/phone_gen/generate.py" "dark castle" --cfg 2.0 --neg "blurry"
python3 "$SDXL_QNN_BASE/phone_gen/generate.py" "landscape" --seed 777 --steps 8
python3 "$SDXL_QNN_BASE/phone_gen/generate.py" "1girl, upper body, looking at viewer, masterpiece, best quality" --seed 777 --steps 8 --cfg 3.5 --prog-cfg
```

Сейчас runtime по умолчанию включает:

- `SDXL_QNN_USE_MMAP=1`
- `SDXL_QNN_PERF_PROFILE=sustained_high_performance`
- live-логирование CPU / GPU / NPU при `SDXL_SHOW_TEMP=1`

Если в phone-side папке одновременно существуют `htp_backend_extensions_lightning.json` и `lib/libQnnHtpNetRunExtensions.so`, `phone_generate.py` теперь автоматически подхватывает `SDXL_QNN_CONFIG_FILE` даже для прямых запусков из Termux.

#### Через APK

```bash
cd APK
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

APK даёт полноценный GUI: промпт, негативный промпт, CFG, steps, seed, контрастирование, прогресс-бар, live-температуры CPU / GPU / NPU и сохранение в галерею.  
В `v0.2.3` APK доступны опциональные переключатели **Live Preview (TAESD)** и **½-CFG**, запуск phone runtime по умолчанию включает QNN `mmap` + `sustained_high_performance`, при наличии нужных `.json` + `.so` автоматически прокидывается backend-extension config, временные runtime-файлы пишутся в app-private cache вместо общей папки, APK корректно парсит preview-тайминги вида `QNN GPU ...ms`, а документация уже описывает rebuilt QNN TAESD preview path на GPU с реальным временем порядка **1.0 s** на шаг.
Текущий путь по умолчанию — `/sdcard/Download/sdxl_qnn`; через ⚙️ Settings можно указать другую раскладку.

#### Host-side (с ПК через ADB)

```bash
python SDXL/generate.py "cat on windowsill, masterpiece" --seed 42
```

## Что реально лежит на телефоне сейчас?

Текущая основная цель для деплоя — `/sdcard/Download/sdxl_qnn`, но живой слепок с телефона по ссылке ниже остаётся полезным как исторический пример более ранней rooted-раскладки.

- минимально необходимая структура — ниже;
- живая историческая структура — в [`examples/phone-sdxl-qnn-layout.md`](examples/phone-sdxl-qnn-layout.md).

## Структура проекта

```text
├── README.md                 ← стартовая страница с выбором языка
├── README_RU.md              ← вы здесь
├── README_EN.md              ← English version
├── LICENSE                   ← Apache 2.0
├── .gitattributes
├── .gitignore
├── phone_generate.py         ← standalone-генератор для телефона
├── tokenizer/                ← BPE токенизатор (CLIP)
│   ├── vocab.json
│   └── merges.txt
├── examples/
│   ├── phone-sdxl-qnn-layout.md    ← живой пример rooted-раскладки на телефоне
│   ├── phone-sdxl-qnn-layout_RU.md ← русская версия примера rooted-раскладки
│   └── rooted-phone-sample/        ← небольшой набор rooted-артефактов (доки, PNG, конфиги, скрипты)
├── .github/
│   └── ISSUE_TEMPLATE/
│       └── bug_report.md
├── scripts/
│   ├── deploy_to_phone.py    ← деплой на телефон через ADB
│   ├── download_qualcomm_sdk.py
│   ├── download_adb.py
│   └── build_all.py          ← ранний SDXL helper (поздние шаги перепроверяются)
├── SDXL/                     ← текущие SDXL-специфичные скрипты конвертации и сборки
│   ├── generate.py           ← host-side генератор (с ПК)
│   ├── bake_lora_into_unet.py
│   ├── export_clip_vae_to_onnx.py
│   ├── export_sdxl_to_onnx.py
│   ├── convert_clip_vae_to_qnn.py
│   ├── convert_lightning_to_qnn.py
│   ├── export_taesd_to_onnx.py
│   ├── convert_taesd_to_qnn.py
│   ├── build_android_model_lib_windows.py
│   ├── assess_generated_image.py
│   ├── verify_clip_vae_onnx.py
│   ├── verify_e2e_onnx.py
│   ├── debug/               ← сюда перенесены лабораторные/диагностические/экспериментальные скрипты
│   ├── LESSONS_LEARNED.md    ← подводные камни и решения
│   └── LESSONS_LEARNED_RU.md ← русская версия lessons learned
└── APK/                      ← Android-приложение
    ├── README.md
    └── app/src/main/
        ├── AndroidManifest.xml
        └── java/com/sdxlnpu/app/
            ├── MainActivity.java
            └── SettingsActivity.java
```

## Архитектура

```text
          ┌──────────────────────────────────────────────────────────────┐
          │                    Телефон (NPU)                            │
          │                                                             │
Prompt ──▶│ CLIP-L ──┐                                                  │
          │ (FP16)   ├──▶ concat [1,77,2048] ──▶ Split UNet ──▶ VAE ──▶│──▶ PNG
          │ CLIP-G ──┘    + pooled [1,1280]      encoder FP16   FP16   │
          │ (FP16)        + time_ids [1,6]       decoder FP16          │
          │                                      × 8 шагов             │
          └──────────────────────────────────────────────────────────────┘
```

**Split UNet:** Полный FP16 UNet (~5 GB) превышает лимит HTP (~3.5 GB), поэтому он разделяется на encoder (conv_in + down_blocks + mid_block, 2.52 GB) и decoder (up_blocks + conv_out, 2.69 GB). Encoder передаёт decoder 11 skip-connections + mid + temb.

**Scheduler:** EulerDiscrete, trailing spacing (требование Lightning), pure numpy.

**Tokenizer:** Pure Python BPE (без HuggingFace/transformers), идентичный CLIP tokenizer.

## Минимальная структура файлов на телефоне

```text
/sdcard/Download/sdxl_qnn/
├── context/                               (QNN context binaries)
│   ├── clip_l.serialized.bin.bin          (~223 MB)
│   ├── clip_g.serialized.bin.bin          (~1.3 GB)
│   ├── unet_encoder_fp16.serialized.bin.bin (~2.3 GB)
│   ├── unet_decoder_fp16.serialized.bin.bin (~2.5 GB)
│   ├── vae_decoder.serialized.bin.bin     (~151 MB)
│   └── taesd_decoder.serialized.bin.bin   (~5-15 MB, optional QNN live preview)
├── htp_backend_extensions_lightning.json  (опциональная точка входа для HTP backend extensions)
├── htp_backend_ext_config_lightning.json  (опциональная tuning-конфигурация HTP backend)
├── phone_gen/
│   ├── generate.py                        (standalone генератор)
│   ├── taesd_decoder.onnx                 (~5 MB, опциональный CPU fallback preview)
│   └── tokenizer/
│       ├── vocab.json                     (CLIP BPE vocabulary)
│       └── merges.txt                     (BPE merge rules)
├── lib/                                   (QNN runtime библиотеки)
│   └── libQnnHtpNetRunExtensions.so       (опционально, auto-used при наличии)
│   └── libQnnGpu.so                       (опционально, для QNN GPU TAESD preview)
├── model/                                 (опционально: дополнительные model libs для некоторых сценариев)
│   └── libTAESDDecoder.so                 (опционально, TAESD QNN preview model fallback)
├── bin/
│   ├── qnn-net-run                        (QNN inference runner)
│   └── qnn-gpu-target-server              (опционально, рекомендуется для QNN GPU preview)
└── outputs/                               (PNG результаты)
```

## Ограничения

- **Разрешение фиксировано** 1024×1024 — другие размеры требуют полной переконвертации
- **Быстрый документированный путь предполагает, что Lightning LoRA уже замержена в UNet** — без этого merge вы фактически уходите в сильно более медленный базовый SDXL path, и тайминги/примеры из репозитория перестают быть репрезентативными
- **VAE FP16** слегка сжимает цветовой диапазон -> применяется percentile contrast stretching
- **TAESD live preview опционален** — runtime теперь в первую очередь пытается использовать задеплоенный QNN TAESD preview path (предпочтительно через GPU backend), а при отсутствии или ошибке QNN-preview assets откатывается на маленький ONNX-декодер (`phone_gen/taesd_decoder.onnx`) плюс `onnxruntime`
- **Лучший текущий быстрый путь использует HTP backend extensions** — JSON-конфиг уже лежит в репозитории, но runtime включает его автоматически только если на телефон реально задеплоен `libQnnHtpNetRunExtensions.so` в `lib/`
- **CFG > 1.0 здесь дорогой** — нужны и conditional, и unconditional предсказания; поскольку runtime использует split UNet (`encoder` + `decoder`), наивный CFG превращает каждый шаг в четыре phone-side запуска UNet-подпроцессов. Текущий runtime уже батчит часть этой работы лучше, чем раньше, но по wall-clock времени это всё равно почти 2× относительно no-CFG пути.
- **Termux обязателен** — Python runtime для `phone_generate.py`
- **Для APK на Android 11+ может понадобиться доступ ко всем файлам** — чтобы читать `/sdcard/Download/sdxl_qnn`
- Тестировалось только на **OnePlus 13 (SM8750)**

## Known issues

- Первый запуск каждого компонента медленнее (загрузка context binary в NPU)
- При низком RAM телефон может убить процесс — закройте другие приложения
- На Android 11+ APK может попросить доступ ко всем файлам для работы с `/sdcard/Download/sdxl_qnn`
- Если `python3` недоступен из процесса приложения, укажите корректную команду/путь в ⚙️ Settings
- Если backend extensions не задеплоены, runtime всё равно будет работать — просто откатится к обычному non-config QNN path
- numpy и torch используют разные RNG — одинаковый seed даёт разные, но валидные изображения

## Лицензия

Apache 2.0 — см. [LICENSE](LICENSE)

Зависимости:

- Qualcomm QAIRT SDK — проприетарная лицензия Qualcomm
- SDXL-Lightning LoRA (ByteDance) — Apache 2.0
- Stable Diffusion XL — CreativeML Open RAIL-M
