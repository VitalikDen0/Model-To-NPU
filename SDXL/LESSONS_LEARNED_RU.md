# SDXL на Qualcomm NPU — уроки, подводные камни и практические выводы

Этот файл — русскоязычный аналог `LESSONS_LEARNED.md`.
Он собран из реальной практики запуска нативного SDXL pipeline на Snapdragon 8 Elite (SM8750) / Hexagon HTP.

> Если вы только заходите в проект, лучше сначала прочитать именно этот файл, а уже потом трогать экспорт, QNN-конвертацию и phone-side runtime.

---

## 1. Подводные камни экспорта в ONNX

### CLIP должен брать `hidden_states[-2]`, а не `last_hidden_state`

Это одна из самых дорогих по времени ошибок.

SDXL использует **предпоследний слой** hidden states у обоих text encoder'ов:

```python
out = text_encoder(ids, output_hidden_states=True)
hidden = out.hidden_states[-2]
# а не out.last_hidden_state
```

- **CLIP-L** имеет 13 слоёв (0–12). Последний слой в ряде прогонов давал `NaN` даже в FP32.
  Рабочим оказался именно слой `hidden_states[-2]`.
- **CLIP-G** использует тот же принцип.
- **Pooled output** нужен только от CLIP-G: `text_embeds -> [1, 1280]`.
- Финальные `prompt_embeds` — это `concat(clip_l_hidden[-2], clip_g_hidden[-2], dim=-1)` → `[1, 77, 2048]`.

### CLIP-G упирается в protobuf > 2 GB

Отсюда три практических правила:

1. сохранять ONNX с `save_as_external_data=True`;
2. проверять модель через `onnx.checker.check_model(path_str)`;
3. для просмотра метаданных открывать ONNX с `load_external_data=False`.

### `input_ids` должны быть `int64` в ONNX, но в runtime их нужно кормить как `float32`

Это неприятный, но важный нюанс.

- На уровне ONNX `input_ids` — это `int64`.
- В QNN runtime сырые входы читаются как байты, и на практике токены нужно подавать как **float32-значения**, а не как int32/int64 byte pattern.

Если подать не тот формат, CLIP не падает сразу, а начинает выдавать мусорные embeddings. Это особенно коварно.

### Прогон FP16 → FP32 → FP16 допустим

Исходные веса здесь FP16, включая VAE.
Временный экспорт через FP32 не ломает численную корректность.

### VAE требует переписывания `InstanceNorm` в `GroupNorm`

QNN не поддерживает `InstanceNorm` так, как это нужно для VAE.
Поэтому `rewrite_onnx_instancenorm_to_groupnorm.py` — не «дополнительный» скрипт, а фактически часть рабочего пайплайна для VAE.

---

## 2. Подводные камни QNN-конвертации

### QAIRT 2.31 и рядом требует monkey-patch'ей

`qnn_onnx_converter_expanddims_patch.py` здесь не декоративный.
Он закрывает целую пачку багов конвертера:

- `ExpandDims`
- смешанные FP16/FP32 elementwise цепочки
- `GroupNorm`
- сохранение dtype в `Reshape`, `LayerNorm`, `Transpose`, `MatMul`, `Softmax`, `Concat`, `Resize` и не только

Именно поэтому удалять такие workaround-скрипты «для чистоты» пока рано.

### Конвертер может сохранять C++ без `.cpp`

QNN converter иногда сохраняет output как файл без расширения.
Сборка Android model libs ожидает `.cpp`, поэтому `build_android_model_lib_windows.py` фактически ещё и чинит артефакты конвертера.

### Калибровочные данные обязаны быть `float32`

Если подать туда float16 `.raw`, можно получить «как будто всё прошло», но квантование станет некорректным.
Это типичный тихий сбой.

### Для cross-attention нужен transpose `NCF -> NFC`

`encoder_hidden_states` после CLIP нужно подавать в том виде, в котором QNN их реально ожидает.
Если забыть transpose, итог часто выглядит как:

- формально корректный прогон;
- картинка с правильным диапазоном;
- но почти без нормального следования промпту.

---

## 3. Phone-side подводные камни

### `ADSP_LIBRARY_PATH` обязателен

Без него `qnn-net-run` и `qnn-context-binary-generator` часто падают с ошибками создания устройства.

Рабочий шаблон:

```bash
export ADSP_LIBRARY_PATH='/sdcard/Download/sdxl_qnn/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp'
```

### Полный UNет нельзя считать «просто моделькой» для `qnn-net-run --model`

На практике для полноразмерного UNet рабочая схема — это `--retrieve_context` с заранее подготовленным context binary.

### Генерация context binary на телефоне — тяжёлая операция

Она может занимать минуты и сильно грузить телефон.
Если выставить слишком агрессивный backend-профиль, устройство становится нестабильным.

### Внутреннее имя графа у folderized converter output — `model`

Это очень неочевидный практический нюанс.
Если положиться на имя каталога, `composeGraphs()` может падать.

### В QNN spatial tensors нужно кормить в NHWC

Если где-то по привычке оставить NCHW — можно получить тихо некорректный результат без понятной ошибки.

---

## 4. Особенности именно SDXL-Lightning

### Lightning не любит классический CFG как обычный SDXL

Это distilled-модель, так что привычные представления о CFG тут надо применять аккуратно.

### Почему CFG здесь так сильно бьёт по wall-clock времени

В этом репозитории CFG дорогой сразу по двум причинам:

1. classifier-free guidance требует и **cond**, и **uncond** предсказания;
2. phone-side runtime использует **split UNet** (`encoder` + `decoder`).

То есть наивно один шаг CFG превращается не просто в «два UNet-прогона», а в:

- uncond encoder
- uncond decoder
- cond encoder
- cond decoder

Итог: один denoising step легко распухает до **четырёх QNN-подпроцессов** плюс дополнительного file I/O.
Текущий phone runtime уже умеет батчить cond/uncond часть этой работы заметно лучше, чем раньше, но реальной NPU-работы всё равно примерно вдвое больше, чем в no-CFG пути.

### Scheduler должен быть `EulerDiscreteScheduler` c `timestep_spacing="trailing"`

Это не эстетика, а рабочее требование Lightning.
С «leading» spacing результаты уходят в неверные timesteps.

### LoRA нужно мерджить заранее

Lightning здесь приходит как LoRA, а не как готовый monolithic UNet для NPU runtime.
Поэтому `bake_lora_into_unet.py` — обязательная стадия.

---

## 5. Что выяснилось по mixed precision и квантованию

### Глобальный FP16 override на все LayerNorm + Softmax ломает конвертер

Это выглядит как магия, но воспроизводится.

Симптом:

- шкала выхода резко схлопывается;
- UNet перестаёт давать адекватный noise prediction.

Локальные выборочные override'ы могут работать, а полный глобальный набор — нет.

### Имена override'ов должны совпадать именно с ONNX source names

Не с санитизированными IR-именами, а с исходными именами тензоров из ONNX.
Иначе QAIRT просто молча проигнорирует настройку.

### `Convert islands` в некоторых mixed-precision сценариях ломают phone-side ctxgen

Даже если на хосте всё выглядит обещающе, context binary generation на телефоне может стабильно повисать по таймауту.

---

## 6. Сводка по dtype весов

| Компонент | dtype | Примечание |
| --------- | ----- | ---------- |
| CLIP-L | `torch.float16` | рабочий экспорт подтверждён |
| CLIP-G | `torch.float16` | рабочий экспорт подтверждён |
| UNet | `torch.float16` | после merge Lightning LoRA |
| VAE | `torch.float16` | не BF16 |

---

## 7. Проверенная численная близость

На текущем этапе были подтверждены:

- parity между PyTorch и ONNX для CLIP-L;
- parity для CLIP-G;
- parity для `prompt_embeds`;
- хорошая близость для VAE decoder.

Это значит, что основные проблемы проекта сейчас лежат не в «сломанных экспортёрах как таковых», а в:

- QNN conversion edge cases;
- runtime-layout нюансах;
- phone-side deployment и совместимости окружения.

---

## 8. Практический статус набора скриптов

На сегодня скрипты удобно делить так:

### Ядро публичного пайплайна

- `convert_sdxl_checkpoint_to_diffusers.py`
- `bake_lora_into_unet.py`
- `export_clip_vae_to_onnx.py`
- `export_sdxl_to_onnx.py`
- `convert_clip_vae_to_qnn.py`
- `convert_lightning_to_qnn.py`
- `build_android_model_lib_windows.py`
- `scripts/deploy_to_phone.py`
- `phone_generate.py`

### Проверка и диагностика

- `verify_clip_vae_onnx.py`
- `verify_e2e_onnx.py`
- `assess_generated_image.py`
- `debug/measure_ram.py`

### Экспериментальные / продвинутые

- `debug/quantize_unet.py`
- `debug/run_full_phone_pipeline.py`
- `debug/run_ctxgen_lightning.sh`
- calibration-скрипты
- ONNX/QAIRT workaround-скрипты

То есть «лишних» файлов в явном виде почти нет, но **не все они обязательны для первого знакомства с проектом**.

---

## 9. Чего здесь пока не хватает

Файл уже полезный, но не финальный. В перспективе сюда стоит добавить:

1. отдельный раздел по **сравнению rooted и non-root layout**;
2. таблицу **какие скрипты обязательны для минимального happy-path**, а какие только для исследований;
3. раздел по **типичным сбоям APK/Termux integration**;
4. отдельный блок по **публичному beta-release процессу**.

---

## 10. Вывод

Если коротко, главные реальные выводы проекта такие:

- SDXL на Snapdragon NPU — не «демо на бумаге», а рабочий pipeline;
- основная сложность не в одной большой модели, а в сумме маленьких несовместимостей между PyTorch, ONNX, QAIRT, QNN runtime и Android;
- workaround'и в таком проекте — не мусор, а накопленное техническое знание;
- публичность проекту не мешает, если честно помечать, что часть пайплайна ещё экспериментальная.

---

## 11. Что показал разбор runtime-overhead (2026-03-30)

Глубокий probe split-UNet пути показал, что оставшаяся боль — это уже не только чистая математика на ускорителе.

- single encoder wall time: **8312.0 ms**;
- single decoder wall time: **8141.1 ms**;
- добавление `--use_mmap` уменьшило эти значения до **6115.0 ms** и **6011.4 ms** соответственно (около **26%** выигрыша на обеих половинах);
- repeat×4 в одном процессе сжал среднюю стоимость до **3527.5 ms** (encoder) и **4379.7 ms** (decoder), а это уже очень явно указывает на overhead жизненного цикла процесса/context.

Практический вывод:

- в `v0.1.3` `mmap` теперь включён по умолчанию и в phone runtime, и в APK launch-path;
- следующий большой speed-win вероятнее всего даст persistent encoder/decoder runner, а не очередной заход в квантование.

---

## 12. Реальный root cause TAESD QNN preview и его фиксация (2026-04-01)

Сломанный TAESD preview оказался проблемой в несколько слоёв.

### Часть 1: путь сохранения preview был слишком наивным

`phone_generate.py` всегда применял `img / 2 + 0.5` перед сохранением preview PNG.
Это корректно только тогда, когда входящий тензор живёт примерно в диапазоне `[-1, 1]`.

В phone-side QNN TAESD пути бывали случаи, когда выход уже приходил в `[0, 1]`, и тогда такой save-path выбеливал картинку примерно в диапазон `[0.5, 1.0]`.

### Часть 2: более серьёзная реальная проблема была в stale / bad TAESD QNN артефактах

Даже после фикса preview-нормализации задеплоенный QNN TAESD preview всё ещё сильно расходился с ONNX-референсом.

Что раскрыло настоящий источник проблемы:

- ONNX TAESD vs финальный VAE image: около **0.78** корреляции на одном и том же latent;
- старый QNN TAESD vs ONNX TAESD: всего около **0.21** корреляции;
- старый raw output у QNN TAESD был зажат в **`[0, 1]`**, тогда как ожидаемый диапазон был примерно **`[-1.18, 1.23]`**.

### Что реально починило ситуацию

Надёжный фикс выглядел так:

1. пересобрать `libTAESDDecoder.so` из текущего ONNX;
2. генерировать TAESD GPU context **прямо на телефоне** через `qnn-context-binary-generator`;
3. деплоить именно эту rebuilt-пару `lib + context` вместе.

После rebuild'а:

- rebuilt QNN TAESD vs ONNX TAESD подскочил примерно до **0.9999** корреляции;
- rebuilt QNN TAESD vs финальная картинка на том же latent дошёл примерно до **0.91**;
- реальная похожесть финального 8-step preview к итоговой картинке на телефоне поднялась примерно до **0.85**.

### Практический вывод для деплоя

Нельзя считать старую пару TAESD `lib/context` «заведомо рабочей» только потому, что preview backend стартует и печатает тайминги.
Если preview выглядит сломанным, нужно сравнивать сам QNN output напрямую с ONNX TAESD reference на одном и том же latent.

Preview вполне может быть *быстрым* и при этом *численно неправильным*.

---

Если вы обновляете пайплайн, переносите его на другой Snapdragon-чип или пробуете другую модельную семью, лучше сначала обновить этот файл, а уже потом код.
