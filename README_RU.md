# Model-to-NPU Pipeline for Snapdragon

**Языки:** [English](README_EN.md) | [Русский](README_RU.md)

> **SDXL на Snapdragon 8 Elite NPU — ~30 с итого** (UNet ~19 с, VAE ~1.9 с, CLIP ~9 мс кэш)
> при 1024×1024, 8 шагов, CFG=3.5, progressive guidance.

> [!TIP]
> End-to-end SDXL flow доступен и практически подтверждён (`checkpoint -> итоговый PNG на телефоне`).
> Начата работа над **SD3**, **Flux**, **Wan** и другими семействами моделей — они будут выходить по мере того, как будут разработаны и проверены методы оптимизации.

<p align="center">
  <b>Репозиторий для model-to-NPU pipeline'ов на Qualcomm Snapdragon</b><br>
  Текущий реализованный pipeline: <b>SDXL на Qualcomm Hexagon NPU</b>.
</p>

---

## Что это?

Этот репозиторий задуман как общее место для нескольких **model-specific pipeline'ов**, нацеленных на Qualcomm Snapdragon NPU.

- для каждой семьи моделей предполагается своя папка;
- **текущая реализованная папка** — `SDXL/`;
- ранний исследовательский workspace `WAN 2.1 1.3B/` выделен под Wan 2.1 T2V 1.3B;
- начата работа над **SD3**, **Flux**, **Wan** и другими — по мере проработки методов оптимизации;
- **текущее Android-приложение** лежит в `APK/`;
- общие deploy-скрипты и файлы — в `scripts/`, `tokenizer/` и в корне.

Сейчас реально реализованный и задокументированный путь — это **Stable Diffusion XL**, работающий **нативно на NPU телефона** (Hexagon HTP). Текущий SDXL pipeline использует CLIP-L, CLIP-G, Split UNet (encoder + decoder) и VAE прямо на устройстве.

**Текущее протестированное сочетание:** [WAI Illustrious SDXL v1.60](https://civitai.com/models/827184/wai-illustrious-sdxl?modelVersionId=2514310) + [SDXL-Lightning 8-step LoRA](https://huggingface.co/ByteDance/SDXL-Lightning) (ByteDance)

> **Важно по скорости:** публичные тайминги и примеры предполагают, что **Lightning LoRA уже замержена в UNet**.
> **Важно по разрешению:** экспорты, context binaries и примеры рассчитаны на **1024×1024**.

## Текущий статус

- **Направление репозитория:** multi-model pipeline'ы под Snapdragon NPU (SDXL, SD3, Flux, Wan, ...)
- **Сейчас реализованная семья:** `SDXL/`
- **Исследовательский Wan-workspace:** `WAN 2.1 1.3B/`
- **Текущая цель APK:** SDXL
- **Статус скриптов:** практический SDXL цикл (checkpoint → image) подтверждён
- **Статус документации:** обновлена до текущего состояния

## Требования для текущего SDXL pipeline

### Телефон

| Компонент | Требование |
| --------- | ---------- |
| **SoC** | Qualcomm Snapdragon 8 Elite (SM8750) или совместимый с QNN HTP |
| **RAM** | 16 GB (пик ~12 GB, свободно >= 6 GB) |
| **Хранилище** | ~10 GB для моделей и context binary |
| **Root** | Не требуется |
| **Termux** | Python 3.13+, numpy, Pillow, `termux-setup-storage` |

### ПК (для сборки)

| Компонент | Версия |
| --------- | ------ |
| Python | 3.10.x (именно 3.10, не 3.11+) |
| QAIRT SDK | 2.31+ (Qualcomm AI Engine Direct) |
| Android NDK | r26+ (для `.so` и `qnn-multi-context-server`) |
| PyTorch | 2.x |
| Windows | 10/11 |

## Производительность

Замерено на OnePlus 13 (Snapdragon 8 Elite, 16 GB RAM):

### Текущая линия APK (v0.4.3) — обновлённый hotfix самодостаточного runtime

Текущая публичная линия `v0.4.3` сохраняет shared FIFO-backed prewarm server из раннего обновления `v0.4.3`, но также включает follow-up hotfix, который закрывает тот практический провал, из-за которого эта сборка могла ощущаться хуже `v0.4.2`. Явно экспортированный bundled runtime теперь имеет приоритет над устаревшими QNN-файлами в `/data/local/tmp`, bundled backend-extension config с относительными путями больше не используется как есть и корректно перестейдживается, а сам APK теперь действительно пакует недостающие core runtime-части (`qnn-net-run` и нужные HTP/System библиотеки) для быстрого пути. Со стороны UI жёстче очищается lifecycle preview/final bitmap'ов и старых preview poller'ов, а `832x480` больше не показывается как общий SDXL preset — он оставлен только для WAN-ветки, где ему и место. Последний refresh также уводит decode preview PNG с UI thread, при включённом live preview включает stride=4, ретраит transient `BlockingIOError` / `InterruptedError` при чтении shared FIFO и перестаёт размножать helper-процессы в hot-load пути за счёт дедупликации queued prewarm launch и запуска Python через `exec`, чтобы приложение отслеживало реальный helper, а не короткоживущую shell-обёртку.

### v0.4.0 — Переменное разрешение + автономный APK

Поддержка переменного разрешения (512×512 до 1536×1536, любое кратное 8). Каталоги QNN-контекстов по разрешению. Выбор разрешения в APK. `build_termux_prefix.py` для извлечения standalone-префикса.

### v0.3.0 — Persistent multi-context server

`seed=44`, `steps=8`, `CFG=3.5`, `--prog-cfg`, Live Preview OFF:

| Этап | Время | Примечания |
| ---- | ----- | ---------- |
| CLIP-L + CLIP-G | ~9 мс | кэшированный результат (первый запуск ~2.8 с) |
| UNet (8 шагов) | ~19.3 с | ~2411 мс/шаг через persistent server + RUN_CHAIN |
| VAE decoder | ~1.9 с | FP16 |
| **Итого (тёплый)** | **~30.4 с** | |

Пик RAM: **~12 GB** из 16 GB
Разрешение: **1024×1024** (фиксированное)

### Предыдущие версии

| Версия | Итого | UNet | CLIP | VAE | Примечания |
| ------ | ----- | ---- | ---- | --- | ---------- |
| v0.2.5 | 75.6 с | 66.6 с | 2.8 с | 3.0 с | burst + native accel, per-step qnn-net-run |
| v0.2.0 | 79.7 с | 72.4 с | 2.9 с | 3.4 с | sustained_high_performance |
| v0.1.3 | 104.4 с | 91.5 с | 2.0 с | 9.0 с | mmap включён |
| v0.1.0 | 273.6 с | — | — | — | первый публичный скриншот |

Подробные исторические данные и заархивированные эксперименты — см. [HISTORY_RU.md](HISTORY_RU.md).

## Как было достигнуто ~19 с на UNet — подробный разбор оптимизаций

UNet ускорился с **66.6 с** (v0.2.5) до **~19.3 с** (v0.3.0) — **ускорение в 3.4 раза**. Вот что именно изменилось и почему каждый элемент важен.

### 1. Persistent multi-context QNN server (самый большой выигрыш)

**Было (v0.2.5):** каждый шаг UNet порождал новый процесс `qnn-net-run`. Каждый процесс:

- `fork()+exec()` — overhead создания процесса (~15–30 мс каждый);
- `dlopen()` библиотек QNN backend каждый раз заново;
- десериализация context binary с диска (~1–3 с на контекст при первой загрузке);
- аллокация и регистрация `rpcmem` shared DSP memory;
- выполнение графа;
- полная очистка и выход.

При 8 шагах × 2 контекста (encoder + decoder) = **16 процессов на изображение** — кумулятивный overhead был огромным.

**Стало (v0.3.0):** один **persistent C-процесс** (`qnn-multi-context-server`) запускается один раз, загружает все context binary и держит их живыми. Он использует простой stdin/stdout протокол:

```text
LOAD <id> <path>     → OK <graph> <inputs> <outputs>
RUN <id> <inputs> <outdir>  → OK <ms>
RUN_CHAIN <enc> <dec> ...   → OK <ms>
QUIT                        → OK
```

Сервер загружает контексты один раз при старте, аллоцирует `rpcmem` один раз, и все последующие запуски графов пропускают весь жизненный цикл процесса. Это одно изменение убрало ~47 с чистого overhead.

### 2. RUN_CHAIN — передача encoder→decoder в памяти

**Было:** после encoder его 11 skip-connection выходов (~82.5 MB итого) записывались на диск как raw файлы, затем decoder-процесс считывал их обратно. Это означало ~165 MB дискового I/O на каждый шаг.

**Стало:** команда `RUN_CHAIN` запускает encoder и decoder последовательно внутри одного серверного процесса. Skip connections передаются через `memcpy` между выходными буферами encoder и входными буферами decoder — **никакого промежуточного файлового I/O**. 11 skip connections + mid + temb остаются в `rpcmem`-буферах сервера.

Примечание: zero-copy pointer swap был опробован, но QNN HTP требует зарегистрированных `rpcmem` handles для каждого буфера, поэтому `memcpy` — минимально возможный подход (см. [HISTORY_RU.md](HISTORY_RU.md)).

### 3. FLOAT_32 direct fread

**Было:** тензорные выходы `qnn-net-run` записывались как текст (один float на строку). Python парсил каждую строку через `float()`.

**Стало:** сервер записывает выходные тензоры как сырой бинарный формат (little-endian float32). Python считывает их одним вызовом `numpy.fromfile`.

### 4. Eager preload — параллельная загрузка контекстов

**Было:** контексты загружались последовательно: CLIP, затем UNet encoder, затем decoder. Контексты UNet вместе занимали ~8 с.

**Стало:** фоновый поток (`_eager_preload_unet()`) отправляет команды `LOAD` для контекстов UNet **пока ещё работает CLIP**. К моменту, когда CLIP завершается, оба UNet-контекста уже прогреты в сервере. Это перекрывает ~8 с загрузки контекстов.

### 5. Кэширование результатов CLIP

**Было:** каждый запуск генерации токенизировал промпт и прогонял CLIP-L + CLIP-G через QNN (~2.8 с).

**Стало:** результаты CLIP кэшируются на диск по хэшу промпта. На cache hit CLIP завершается за ~9 мс (только чтение файлов).

### 6. QNN burst mode + native runtime accelerator

Уже были в v0.2.5, но продолжают вносить вклад:

- **Burst mode:** максимальная частота HTP для коротких интенсивных нагрузок.
- **Native C accelerator:** `libsdxl_runtime_accel.so` ускоряет scheduler math и tensor layout операции.

### Анализ теоретического минимума

При текущей архитектуре:

- **UNet compute:** ~2411 мс/шаг × 8 шагов = ~19.3 с — это реальное время NPU silicon, не может быть сокращено без уменьшения числа шагов или более быстрого чипа.
- **VAE:** ~1.9 с — уже близко к оптимуму.
- **CLIP:** ~9 мс кэш, ~2.8 с холодный — кэшированный практически бесплатен.
- **Overhead сервера на шаг:** ~5–10 мс (memcpy + протокол) — пренебрежимо.
- **Теоретический минимум тёплого прогона:** ~21–22 с (UNet + VAE + минимальная оркестрация).
- **Текущий реальный:** ~30.4 с — оставшиеся ~9 с — Python оркестрация, numpy scheduler math, запись PNG.

## Архитектура

```text
          ┌──────────────────────────────────────────────────────────────────┐
          │                      Телефон (NPU)                              │
          │                                                                 │
          │  ┌─────────────────────────────────────────────────────────┐    │
          │  │        qnn-multi-context-server (persistent C-процесс)  │    │
          │  │                                                         │    │
Prompt ──▶│  │  CLIP-L ──┐                                             │    │
          │  │  (FP16)   ├──▶ concat [1,77,2048]                       │    │
          │  │  CLIP-G ──┘    + pooled [1,1280]                        │    │
          │  │  (FP16)        + time_ids [1,6]                         │    │
          │  │                    │                                     │    │
          │  │         ┌─────────▼──────────┐                          │    │
          │  │         │  RUN_CHAIN × 8     │                          │    │
          │  │         │  encoder ──memcpy──▶ decoder                   │    │
          │  │         │  (11 skip conns     │                          │    │
          │  │         │   в памяти сервера) │                          │    │
          │  │         └─────────┬──────────┘                          │    │
          │  │                   ▼                                      │    │
          │  │              VAE decoder ──▶ PNG                         │    │
          │  └─────────────────────────────────────────────────────────┘    │
          └──────────────────────────────────────────────────────────────────┘
```

**Split UNet:** Полный FP16 UNet (~5 GB) превышает лимит HTP (~3.5 GB), поэтому разделяется на encoder (conv_in + down_blocks + mid_block, 2.52 GB) и decoder (up_blocks + conv_out, 2.69 GB). Encoder передаёт decoder 11 skip-connections + mid + temb через `memcpy` в серверной памяти (RUN_CHAIN).

**Scheduler:** EulerDiscrete, trailing spacing (требование Lightning), pure numpy.

**Tokenizer:** Pure Python BPE (без HuggingFace/transformers), идентичный CLIP tokenizer.

## Быстрый старт

### 1. Подготовка окружения (ПК)

```bash
pip install torch diffusers transformers safetensors onnx onnxruntime Pillow numpy
python scripts/download_qualcomm_sdk.py
python scripts/download_adb.py
```

### 2. Сборка pipeline

```bash
python scripts/build_all.py --checkpoint path/to/model.safetensors
```

Или end-to-end wrapper:

```powershell
pwsh SDXL/run_end_to_end.ps1 -ContextsDir path/to/context_binaries
```

Пошагово:

```bash
# 1. Конвертировать checkpoint в diffusers
python SDXL/convert_sdxl_checkpoint_to_diffusers.py

# 2. Замержить Lightning LoRA в UNet
python SDXL/bake_lora_into_unet.py

# 3. Экспорт в ONNX
python SDXL/export_clip_vae_to_onnx.py
python SDXL/export_sdxl_to_onnx.py

# 4. Конвертация в QNN
python SDXL/debug/convert_clip_vae_to_qnn.py
python SDXL/debug/convert_lightning_to_qnn.py

# 5. Сборка Android model libraries
python SDXL/debug/build_android_model_lib_windows.py

# 6. Сборка persistent multi-context QNN server
python scripts/build_qnn_multi_context_server.py
```

### 3. Деплой на телефон

```bash
python scripts/deploy_to_phone.py \
  --contexts-dir /path/to/context_binaries \
  --phone-base /sdcard/Download/sdxl_qnn \
  --qnn-lib-dir /path/to/qnn_sdk/lib/aarch64-android \
  --qnn-bin-dir /path/to/qnn_sdk/bin/aarch64-android
```

### 4. Подготовка Termux (на телефоне)

```bash
pkg install python python-numpy python-pillow
termux-setup-storage
```

### 5. Генерация

#### Standalone (в Termux на телефоне)

```bash
export PATH=/data/data/com.termux/files/usr/bin:$PATH
export SDXL_QNN_BASE=/sdcard/Download/sdxl_qnn
python3 "$SDXL_QNN_BASE/phone_gen/generate.py" "1girl, anime, cherry blossoms"
python3 "$SDXL_QNN_BASE/phone_gen/generate.py" "landscape" --seed 777 --steps 8 --cfg 3.5 --prog-cfg
```

Runtime по умолчанию:

- `SDXL_QNN_USE_MMAP=1`
- `SDXL_QNN_PERF_PROFILE=burst`
- persistent `qnn-multi-context-server` с RUN_CHAIN

#### Через APK

```bash
cd APK
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

APK даёт полноценный GUI: промпт, негативный промпт, CFG, steps, seed, контрастирование, прогресс-бар, live температуры CPU / GPU / NPU и сохранение в галерею.

## Минимальная структура файлов на телефоне

```text
/sdcard/Download/sdxl_qnn/
├── context/                               (QNN context binaries)
│   ├── clip_l.serialized.bin.bin          (~223 MB)
│   ├── clip_g.serialized.bin.bin          (~1.3 GB)
│   ├── unet_encoder_fp16.serialized.bin.bin (~2.3 GB)
│   ├── unet_decoder_fp16.serialized.bin.bin (~2.5 GB)
│   └── vae_decoder.serialized.bin.bin     (~151 MB)
├── phone_gen/
│   ├── generate.py                        (standalone генератор)
│   └── tokenizer/
│       ├── vocab.json                     (CLIP BPE vocabulary)
│       └── merges.txt                     (BPE merge rules)
├── lib/                                   (QNN runtime библиотеки)
│   └── libQnnHtpNetRunExtensions.so       (опционально, auto-used при наличии)
├── bin/
│   └── qnn-multi-context-server           (persistent QNN server)
└── outputs/                               (PNG результаты)
```

## Структура проекта

```text
├── README.md                 ← стартовая страница
├── README_RU.md              ← вы здесь
├── README_EN.md              ← English version
├── HISTORY_EN.md             ← исторический архив производительности
├── HISTORY_RU.md             ← исторический архив (русский)
├── LICENSE                   ← PolyForm Noncommercial License 1.0.0
├── NOTICE                    ← обязательные notice / attribution
├── phone_generate.py         ← standalone-генератор для телефона
├── tokenizer/                ← BPE токенизатор (CLIP)
├── examples/                 ← примеры и samples
├── scripts/
│   ├── deploy_to_phone.py
│   ├── build_qnn_multi_context_server.py
│   ├── build_all.py
│   └── ...
├── NPU/
│   ├── qnn_multi_context_server.c  ← исходник persistent сервера (C)
│   └── build/                      ← собранный бинарник
├── SDXL/                     ← SDXL конвертация, сборка, лабораторные скрипты
│   ├── debug/                ← экспериментальные/диагностические скрипты
│   └── ...
├── WAN 2.1 1.3B/             ← Wan 2.1 T2V исследовательский workspace
└── APK/                      ← Android-приложение
```

## Ограничения

- **Разрешение фиксировано** 1024×1024 — другие требуют полной переконвертации
- **Быстрый путь предполагает, что Lightning LoRA замержена в UNet**
- **VAE FP16** слегка сжимает цветовой диапазон → применяется percentile contrast stretching
- **TAESD live preview опционален** — QNN GPU или fallback на ONNX
- **CFG > 1.0 дорогой** — примерно 2× относительно no-CFG пути
- **Termux обязателен** — Python runtime для `phone_generate.py`
- Тестировалось только на **OnePlus 13 (SM8750)**

## Known issues

- Первый запуск каждого компонента медленнее (загрузка контекстов)
- При низком RAM телефон может убить процесс — закройте другие приложения
- На Android 11+ APK может попросить доступ ко всем файлам
- numpy и torch используют разные RNG — одинаковый seed даёт разные, но валидные изображения

## Лицензия

Этот репозиторий распространяется под **PolyForm Noncommercial License
1.0.0** — см. [LICENSE](LICENSE) и [NOTICE](NOTICE).

Коротко:

- использование, изучение, изменение и форки — только в **некоммерческих** целях;
- при распространении — приложить условия PolyForm и строки из [`NOTICE`](NOTICE);
- сторонние зависимости сохраняют свои лицензии.

Зависимости:

- Qualcomm QAIRT SDK — проприетарная лицензия Qualcomm
- SDXL-Lightning LoRA (ByteDance) — Apache 2.0
- Stable Diffusion XL — CreativeML Open RAIL-M
