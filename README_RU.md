# Model-to-NPU Pipeline для Qualcomm Snapdragon

[![Snapdragon 8 Elite](https://img.shields.io/badge/SoC-Snapdragon%208%20Elite%20(SM8750)-red.svg)](https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-elite-mobile-platform)
[![Qualcomm Hexagon](https://img.shields.io/badge/NPU-Hexagon%20V79%20HTP-blue.svg)](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm%20Noncommercial-green.svg)](LICENSE)
[![Engine](https://img.shields.io/badge/Движок-v0.6.0--core%20(Монолитный)-orange.svg)](PROJECT_STATE.md)

**Языки:** [English](README_EN.md) | [Русский](README_RU.md) | [Состояние проекта](PROJECT_STATE.md) | [Android APK](APK/README.md)

Первый в мире полностью рабочий локальный пайплайн **Stable Diffusion XL (SDXL)**, работающий **нативно на мобильном NPU Qualcomm Hexagon** (Snapdragon 8 Elite / OnePlus 13) без облачных серверов, без Termux, без Root-прав и без разделения модели на части.

---

## ⚡ Ключевые прорывы версии v0.6.0-core

### 1. 🚀 Достигнуто 72.7% утилизации физического предела NPU
- **7.13 устойчивых TOPS** по всем 362 слоям SDXL 2.57B (W8A16) на Hexagon V79 HTP.
- Время одного прохода UNet снижено до **880.3 мс** (с исторических ~2411 мс на ранних версиях и ~956 мс до разблокировки HVX).
- Суммарное время 8-шаговой денойз-генерации UNet упало до рекордных **11.59 секунд**!
- Полный Cold Start (включая десериализацию 2.44 ГБ весов из UFS 4.0 flash) — **~20–21 секунда**. Последующие теплые генерации — **~14 секунд**.
- **Оверхед хоста (CPU + память) сведён к < 0.7%**:
  - ARM64 NEON FP16 MLP FMA: 3.4 мс (0.4%)
  - L1-удвоенная запись в ION RPCMEM: 2.2 мс (0.2%, скорость ~49.2 ГБ/с)
  - Квантование и транспонирование: 0.4 мс (0.0%)
  - FastRPC шина: 1.7 мс (0.2%)
- **Полезный цикл конвейера NPU (Pipeline Duty Cycle): 98.7%**!

### 2. 🧩 Монолитный UNet W8A16 (Отказ от деления модели)
- Модель UNet больше **не делится** на encoder и decoder!
- Единый монолитный граф (`unet.serialized.bin`, ~2.44 ГБ) загружается и исполняется как единый контекст.
- Полностью устранены:
  - 11 межпроцессных буферов skip-connections (82.5 МБ на шаг);
  - Накладные расходы IPC и рассинхронизация контекстов;
  - Повторные аллокации памяти DSP.

### 3. 📐 Динамическое разрешение на лету (от 0.26 MP до 2.36 MP)
- **Любые пропорции и разрешения** от 512×512 до 1536×1536 / 1344×1728 без перекомпиляции графов и без переключения контекстов.
- **Метод Centered Spatial-CFG Sub-Canvas Framing**:
  - Активное окно размещается в оптическом центре (512, 512) с пространственным маскированием весов CFG (w_CFG = 3.5 в кадре с гладким спадом до 1.0 на краях).
  - Естественная эволюция полного латента с непрерывной дисперсией сохраняет 100% стабильность GroupNorm (ноль `NaN`).
  - Полное устранение артефактов билатеральной симметрии («зеркала Роршаха») и швов на краях.
  - Встроенный 4×4 Catmull-Rom ресемплинг + Contrast-Adaptive Sharpening (CAS, 59 мс).
- **0% дополнительной нагрузки на NPU** по сравнению со стандартным 1024×1024!

### 4. 📱 Архитектура Zero-Root / Zero-Termux / Zero-Python
- **Root-права НЕ требуются**.
- **Termux и среда Python на телефоне НЕ требуются**.
- Полностью автономный нативный движок на C (`qnn-multi-context-server`) управляет CLIP, монолитным UNet и VAE напрямую через C API QNN.
- Запуск одной командой через ADB или прямо из нативного APK.

### 5. 📲 Готовится обновление Android APK (до версии 0.6.0)
- Нативное Android-приложение в каталоге `APK/` прямо сейчас обновляется до версии **0.6.0**.
- В версии 0.6.0 старый стек на базе Termux/Python полностью заменяется на монолитный C-движок с удобным UI выбора любого разрешения и моментальной генерацией.

---

## 🖼️ Галерея генераций

<!-- markdownlint-disable MD033 -->
<table align="center">
  <tr>
    <td width="50%"><img src="https://github.com/user-attachments/assets/915ef71e-d72b-4fa0-823d-b316289f2041" alt="SDXL on phone sample 1" width="100%"></td>
    <td width="50%"><img src="https://github.com/user-attachments/assets/4bc1ac51-a98e-4931-a3e9-247327e0bbe5" alt="SDXL on phone sample 2" width="100%"></td>
  </tr>
  <tr>
    <td width="50%"><img src="https://github.com/user-attachments/assets/1c87282c-ccc2-4dc1-b003-0693dd0fa3d4" alt="SDXL on phone sample 3" width="100%"></td>
    <td width="50%"><img src="https://github.com/user-attachments/assets/8f5e3d0d-ebe6-4cea-98f7-2b13b51a9ede" alt="SDXL on phone sample 4" width="100%"></td>
  </tr>
</table>
<!-- markdownlint-enable MD033 -->

Все примеры выше — это честные **1024×1024** генерации модели WAI Illustrious + SDXL-Lightning, полученные непосредственно на телефоне.

---

## 📱 Подтверждение работы на телефоне (Proof that it actually runs on-device)

<!-- markdownlint-disable MD033 -->
<table align="center">
  <tr>
    <td width="50%" align="center">
      <b>Ранний публичный скриншот — 273.6 с итого</b><br>
      <img src="https://github.com/user-attachments/assets/15c785f0-b7a3-4dac-8535-e14055bf3453" alt="Earlier phone-side proof screenshot at 273.6 seconds" width="100%">
    </td>
    <td width="50%" align="center">
      <b>Публичный маркер v0.2.0 — 100.8 с итого</b><br>
      <img src="https://github.com/user-attachments/assets/70988ed8-bf42-4235-8a70-19bf35db6574" alt="Phone-side proof screenshot for v0.2.0 at 100.8 seconds" width="100%">
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <b>Скриншот v0.2.3 (Live Preview ON) — 78.0 с итого</b><br>
      <img src="https://github.com/user-attachments/assets/e36a584f-bb39-427a-805d-ea44e9a8b3a0" alt="Phone-side proof screenshot for v0.2.3 at 78.0 seconds" width="100%">
    </td>
    <td width="50%" align="center">
      <b>Cold-start APK замер v0.4.7 — 34.6 с итого</b><br>
      <img src="https://github.com/user-attachments/assets/04b6e61a-79d6-4ce5-a7d6-158461ca97e6" alt="Current phone-side proof screenshot at 34.6 seconds (cold start)" width="100%"><br>
      <sub>Чистое время ускорителя: ~16.25 с.</sub>
    </td>
  </tr>
</table>
<!-- markdownlint-enable MD033 -->

### Эволюция скорости и телеметрия на OnePlus 13

Хронология публичных подтверждений скорости: **273.6 с → 100.8 с → 78.0 с → 34.6 с → 11.59 с (чистый UNet 8 шагов)**!

```text
================================================================================
  OnePlus 13 (Snapdragon 8 Elite / Hexagon V79 HTP) — Лог выполнения на устройстве
================================================================================
  [Init] Загрузка QNN backend + system: 142 мс
  [Load] CLIP-L + CLIP-G загружены за: 312 мс
  [Load] Монолитный UNet W8A16 (2.44 ГБ) загружен за: 5821 мс
  [Load] VAE Decoder FP16 загружен за: 284 мс
  [CLIP] Кодирование промпта: 281 мс
  [Zero-Copy Denoise] Автономный денойзинг 8 шагов в памяти Hexagon NPU...
  [UNet 1/8] 880.3 мс (CFG active)  | 6 потоков HVX @ 7.13 TOPS
  [UNet 2/8] 879.8 мс (CFG active)  | 6 потоков HVX @ 7.13 TOPS
  [UNet 3/8] 878.9 мс (CFG active)  | 6 потоков HVX @ 7.13 TOPS
  [UNet 4/8] 876.5 мс (CFG active)  | 6 потоков HVX @ 7.13 TOPS
  [UNet 5/8] 874.2 мс (CFG dCache)  | 6 потоков HVX @ 7.13 TOPS
  [UNet 6/8] 874.1 мс (CFG dCache)  | 6 потоков HVX @ 7.13 TOPS
  [UNet 7/8] 874.0 мс (CFG dCache)  | 6 потоков HVX @ 7.13 TOPS
  [UNet 8/8] 874.5 мс (CFG dCache)  | 6 потоков HVX @ 7.13 TOPS
  [Zero-Copy Denoise] Итого UNet: 11,590 мс (880.3 мс/проход, NPU Duty Cycle: 98.7%)
  [VAE] Декодирование латентов: 2,204 мс (FP16)
  [Save] Сохранение PNG: 189 мс
  Полное время Cold Start: ~20.9 с | Теплый повторный запуск: ~14.1 с
================================================================================
```

---

## 📊 Таблица бенчмарков и телеметрии (OnePlus 13 / Snapdragon 8 Elite)

### Аппаратный профиль одного прохода UNet (1024×1024)

| Этап конвейера | Вычислительный блок | Задержка | Доля времени | Оптимизация |
| :--- | :--- | :---: | :---: | :--- |
| **Host `temb` MLP** | Oryon CPU (ARM NEON) | 3.43 мс | 0.4% | FP16->FP32 FMA (`vld1q_f16`, `vfmaq_f32`) |
| **Запись Host -> RPCMEM ION** | Системная шина RAM | 2.20 мс | 0.2% | L1-кэшированный блочный трансфер (~49.2 ГБ/с) |
| **Квантование и транспонирование** | Oryon CPU | 0.41 мс | 0.0% | NEON векторизация |
| **FastRPC транспорт** | Шина ARM <-> Hexagon | 1.72 мс | 0.2% | Синхронизация буферов ION |
| **Чистый расчет Hexagon V79** | **Hexagon V79 HTP (NPU)** | **880.31 мс** | **98.9%** | **Разблокировано 6 потоков HVX** |
| **Суммарный проход UNet** | — | **888.07 мс** | **100%** | **Полезный цикл NPU: 98.7%** |

### Физический Roofline NPU Hexagon V79

- **Размер весов монолитного UNet:** 2.44 ГиБ (W8A16).
- **Практическая пропускная способность памяти LPDDR5X (OnePlus 13):** ~58 ГБ/с.
- **Теоретический предел потокового чтения весов из RAM:** 2.44 × 1024 / 58 000 ≈ 43 мс/слой → ~640 мс на весь 362-слойный граф (предел ~9.8 TOPS).
- **Фактическое время Hexagon V79:** **880.3 мс** = **7.13 устойчивых TOPS**.
- **Эффективность утилизации физического предела:** **72.7%**!

### История версий и прогресс скорости

| Версия | Полное время | UNet (8 шагов) | CLIP | VAE | Архитектура и стек |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **v0.1.0** | 273.6 с | — | — | — | Первый публичный запуск на телефоне |
| **v0.1.3** | 104.4 с | 91.5 с | 2.0 с | 9.0 с | Включение mmap |
| **v0.2.0** | 79.7 с | 72.4 с | 2.9 с | 3.4 с | Режим sustained high performance |
| **v0.2.5** | 75.6 с | 66.6 с | 2.8 с | 3.0 с | Нативный ускоритель, вызовы `qnn-net-run` |
| **v0.3.0** | 30.4 с | 19.3 с | 2.8 с | 1.9 с | Persistent C-сервер, деление UNet (enc+dec) |
| **v0.4.7** | 34.6 с | 14.2 с | 0.1 с | 1.8 с | Замер cold start APK (разделенный UNet) |
| **v0.6.0-core** | **20.0–21.0 с** | **11.59 с** | **0.28 с** | **2.20 с** | **Монолитный W8A16, 72.7% NPU Roofline, 6 HVX потоков, Zero-Root/Termux** |

> *Примечание: Время v0.6.0-core (~20–21 с) измерено при холодном старте с полной десериализацией 2.44 ГБ графа из UFS 4.0 flash. Последующие генерации без выгрузки контекста занимают ~14 секунд!*

### Замеры динамического разрешения (Centered Spatial-CFG)

| Разрешение | Соотношение | Мегапиксели | UNet (8 шагов) | Полное время | Качество и особенности |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1024 × 1024** | 1:1 Квадрат | 1.05 MP | **11.59 с** | 21.06 с | Базовый референс |
| **832 × 1216** | 9:16 Портрет | 1.01 MP | **11.64 с** | 20.64 с | Идеальная анатомия, ноль швов |
| **1024 × 768** | 4:3 Пейзаж | 0.79 MP | **11.91 с** | 20.74 с | Высокая детализация ландшафта |
| **1344 × 1728** | 3:4 Hi-Res | 2.32 MP | **11.21 с** | 20.03 с | Высокая резкость Catmull-Rom CAS |

---

## 🛠️ Быстрый старт и запуск

### Требования
- **Устройство**: Смартфон на базе Qualcomm Snapdragon 8 Elite (OnePlus 13 и аналоги) с NPU Hexagon V79.
- **Оперативная память**: 16 ГБ LPDDR5X (пиковое потребление монолитного пайплайна ~3.5 ГБ).
- **Хранилище**: ~8 ГБ на внутренней памяти устройства.
- **Root**: **НЕ ТРЕБУЕТСЯ**.
- **Termux**: **НЕ ТРЕБУЕТСЯ**.

### 1. Сборка нативного C-движка (на ПК)
Требуются Android NDK (r26+) и Qualcomm QAIRT SDK (2.31+):
```bash
python scripts/build_qnn_multi_context_server.py --deploy
```

### 2. Загрузка моделей на телефон
```bash
adb push D:/platform-tools/sdxl_npu/unet.serialized.bin /data/local/tmp/sdxl_qnn/
adb push D:/platform-tools/sdxl_npu/vae_decoder.serialized.bin /data/local/tmp/sdxl_qnn/
adb push D:/platform-tools/sdxl_npu/clip_l.serialized.bin /data/local/tmp/sdxl_qnn/
adb push D:/platform-tools/sdxl_npu/clip_g.serialized.bin /data/local/tmp/sdxl_qnn/
adb push D:/platform-tools/sdxl_npu/tokenizer /data/local/tmp/sdxl_qnn/
```

### 3. Запуск генерации через ADB
```bash
adb shell "LD_LIBRARY_PATH=/data/local/tmp/sdxl_qnn/lib /data/local/tmp/sdxl_qnn/bin/qnn-multi-context-server \
  --backend libQnnHtp.so \
  --system libQnnSystem.so \
  --generate-dyn \
  --prompt 'masterpiece, 1girl, cyberpunk aesthetic, neon city, highly detailed' \
  --width 832 --height 1216 --steps 8 --cfg 3.5 --seed 42 \
  --output /sdcard/Download/output.png"
```

---

## 📁 Структура репозитория

- `NPU/qnn_multi_context_server.c` — Полный автономный нативный движок на C (NEON FMA, RPCMEM ION, 6 HVX потоков, динамический Spatial-CFG).
- `scripts/build_qnn_multi_context_server.py` — Скрипт автоматической сборки под Android NDK Clang.
- `phone_generate.py` — Автономный Python-скрипт для тестов и валидации.
- `PROJECT_STATE.md` — Подробный инженерный паспорт проекта, расчеты физического Roofline и архитектура.
- `APK/` — Исходный код Android-приложения (готовится релиз 0.6.0).
- `SDXL/` — Скрипты квантования, калибровки и сборки ONNX-графов SDXL.
- `WAN 2.1 1.3B/` — Исследовательский workspace под генерацию видео WAN 2.1.

---

## 📜 Лицензия

Проект распространяется под лицензией [PolyForm Noncommercial License 1.0.0](LICENSE).  
Некоммерческое использование, изучение и модификация разрешены.
