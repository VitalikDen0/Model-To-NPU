# SDXL scripts overview

Подробная карта текущего содержимого `SDXL/`.

> [!IMPORTANT]
> В папке `SDXL/` сейчас лежат **не только файлы happy-path**, но и диагностика, проверка численной близости, калибровка, QAIRT/QNN workaround'ы и экспериментальные ветки.
> Это нормально для живого R&D-репозитория.
> [!NOTE]
> После cleanup-реорганизации лабораторные/диагностические/экспериментальные скрипты перенесены в `SDXL/debug/`, а в корне `SDXL/` оставлен самый короткий практический public path.

## Как читать эту карту

- **core-happy-path** — то, что относится к основной документируемой цепочке;
- **deploy-runtime** — запуск/деплой на телефон или Android-специфика;
- **calibration-data** — генерация calibration/raw/input-list файлов;
- **verification-debug** — проверка parity, сравнение, диагностика;
- **experimental-alternative** — исследовательские ветки, не основной путь;
- **utility** — вспомогательные переписчики/оценщики/патчи.

## Две реальные ветки в репозитории

### 1. Публичный beta runtime

Это текущий более понятный путь для чтения репозитория и демонстрации результата:

- `phone_generate.py`
- `scripts/deploy_to_phone.py`
- split-UNet context binaries (`unet_encoder_fp16` + `unet_decoder_fp16`)
- CLIP-L / CLIP-G / VAE context binaries
- APK из `APK/`

Именно эта ветка лучше всего совпадает с текущими `README`.

### 2. Экспериментальная Lightning/QNN lab-ветка

Это ветка активных исследований внутри `SDXL/`:

- `debug/convert_lightning_to_qnn.py`
- `debug/run_phone_lightning.py`
- `debug/run_full_phone_pipeline.py`
- `debug/trace_unet_layer_parity.py`
- `debug/compare_*`
- `debug/generate_*_references.py`
- shell-скрипты для `ctxgen`

Она важна, но **не является самым коротким публичным маршрутом** для первого знакомства с проектом.

## Рекомендуемый минимум для чтения с нуля

Если нужен самый короткий маршрут по репозиторию, начните с:

1. `convert_sdxl_checkpoint_to_diffusers.py`
2. `bake_lora_into_unet.py`
3. `export_clip_vae_to_onnx.py`
4. `export_sdxl_to_onnx.py`
5. `scripts/build_all.py`
6. `scripts/deploy_to_phone.py`
7. `phone_generate.py`
8. `README_RU.md`
9. `SDXL/LESSONS_LEARNED_RU.md`

## Полная инвентаризация Python-файлов

### Core happy-path

| Файл | Что делает | Статус |
| --- | --- | --- |
| `convert_sdxl_checkpoint_to_diffusers.py` | Конвертирует исходный `.safetensors` checkpoint в diffusers-папку. | ✅ Основной шаг |
| `bake_lora_into_unet.py` | Навсегда мерджит SDXL-Lightning LoRA в базовый UNet. | ✅ Основной шаг |
| `export_clip_vae_to_onnx.py` | Экспортирует CLIP-L, CLIP-G и VAE decoder в ONNX. | ✅ Основной шаг |
| `export_sdxl_to_onnx.py` | Экспортирует UNet и связанные SDXL-компоненты в ONNX. | ✅ Основной шаг |
| `debug/convert_clip_vae_to_qnn.py` | Переводит CLIP/VAE ONNX-модели в QNN-артефакты. | ⚠️ Продвинутый/debug-путь |
| `debug/convert_lightning_to_qnn.py` | Переводит Lightning UNet в QNN-модельную цепочку. | ⚠️ Продвинутый/debug-путь |
| `run_end_to_end.ps1` | E2E-обёртка для build/deploy/smoke-пути с управляемыми skip-флагами. | ✅ Основной orchestration helper |
| `debug/generate.py` | Хостовый генератор/оркестратор, который помогает гонять пайплайн через ADB. | ⚠️ Опциональный debug fallback |

### Deploy / runtime

| Файл | Что делает | Статус |
| --- | --- | --- |
| `debug/run_phone_lightning.py` | Запускает phone-side Lightning UNet ветку через ADB и QNN runtime. | ⚠️ Экспериментальный runtime |
| `debug/run_full_phone_pipeline.py` | Гоняет CLIP + UNet + VAE на телефоне как исследовательскую full pipeline ветку. | ⚠️ Экспериментальный runtime |
| `debug/build_android_model_lib_windows.py` | Собирает Android `.so` из QNN model.cpp/model.bin под Windows/NDK. | ⚠️ Важный build-step, но platform-specific |
| `debug/export_split_unet.py` | Экспортирует/подготавливает split UNet для AI Hub и phone-side use. | ⚠️ Альтернативная runtime-ветка |
| `debug/export_and_compile_aihub.py` | Экспорт и компиляция через Qualcomm AI Hub. | ⚠️ Альтернативная облачная ветка |

### Calibration / data prep

| Файл | Что делает | Статус |
| --- | --- | --- |
| `debug/generate_calibration_prompts.py` | Генерирует набор промптов для calibration. | ✅ Вспомогательный, но полезный |
| `debug/make_calibration_data.py` | Делает calibration `.npz` из промптов и диффузионных входов. | ✅ Полезный |
| `debug/make_lightning_calibration.py` | Делает calibration для Lightning с правильным `init_noise_sigma`. | ⚠️ Важный, но ближе к dev-пайплайну |
| `debug/make_qnn_input_list_from_npz.py` | Превращает `.npz` calibration в `input_list.txt` и `.raw` файлы для QNN. | ✅ Полезный |
| `debug/make_qnn_extbias_input_list_from_npz.py` | Строит расширенный input-list для extbias/extmaps сценариев. | ⚠️ Специализированный |

### Verification / debug / parity

| Файл | Что делает | Статус |
| --- | --- | --- |
| `debug/verify_clip_vae_onnx.py` | Сравнивает PyTorch и ONNX для CLIP-L/CLIP-G/VAE. | ✅ Полезная верификация |
| `debug/verify_e2e_onnx.py` | Делает end-to-end sanity-check ONNX-цепочки без phone-side runtime. | ✅ Полезная верификация |
| `debug/verify_vae_quick.py` | Быстрая проверка VAE ONNX. | ⚠️ Локальная техпроверка |
| `debug/compare_unet_pytorch_vs_onnx.py` | Сравнивает выходы PyTorch UNet и ONNX UNet. | ⚠️ Глубокая диагностика |
| `debug/compare_onnx_vs_phone.py` | Сравнивает ONNX-результаты и phone-side результаты. | ⚠️ Глубокая диагностика |
| `debug/batch_compare_onnx_vs_phone_saved_steps.py` | Сравнивает сохранённые шаги ONNX и телефона батчами. | ⚠️ Глубокая диагностика |
| `debug/validate_phone_candidate_with_saved_steps.py` | Валидирует phone-side candidate context на сохранённых шагах и считает parity/perf. | ⚠️ Валидация кандидатов |
| `debug/evaluate_split_unet_8w8a.py` | Оценивает split-UNet 8W8A candidate-профили по качеству/скорости. | ⚠️ Оценка кандидатов |
| `debug/evaluate_unet_8w8a_candidate.py` | Оценивает monolithic UNet 8W8A кандидатов против рабочих baseline'ов. | ⚠️ Оценка кандидатов |
| `debug/host_compare_unet_baselines.py` | Сравнивает несколько host-side baseline UNet цепочек. | ⚠️ Исследовательская диагностика |
| `debug/trace_unet_layer_parity.py` | Трассирует layer-by-layer parity внутри UNet. | ⚠️ Продвинутая диагностика |
| `debug/check_encoder_outputs.py` | Проверяет выходы split encoder против референса. | ⚠️ Внутренняя техпроверка |
| `debug/generate_host_references.py` | Генерирует host-side reference данные для пошагового сравнения. | ⚠️ Исследовательский helper |
| `debug/generate_pc_reference.py` | Делает PC/GPU reference generation для контрольных сравнений. | ⚠️ Исследовательский helper |
| `debug/generate_embed_cfg_references.py` | Строит reference для embedding-space CFG сценариев. | ⚠️ Исследовательский helper |
| `debug/measure_ram.py` | Меряет расход памяти в runtime/phone-side сценариях. | ⚠️ Диагностический helper |
| `debug/sdxl_speed_probe.py` | Гоняет end-to-end замеры скорости на телефоне (и при желании добавляет PC baseline) для текущего runtime-path. | ✅ Runtime diagnostic |
| `debug/sdxl_unet_overhead_probe.py` | Разбирает overhead split-UNet через `qnn-profile-viewer`, включая `mmap`, batched CFG и repeat-в-одном-процессе. | ✅ Runtime diagnostic |

### Utility / rewrite / compatibility

| Файл | Что делает | Статус |
| --- | --- | --- |
| `debug/rewrite_onnx_instancenorm_to_groupnorm.py` | Переписывает `InstanceNorm` в `GroupNorm` для QNN-совместимости. | ✅ Utility (debug-ветка) |
| `debug/rewrite_onnx_shape_reshape_to_static.py` | Делает shape/reshape более статичными под QAIRT. | ✅ Utility (debug-ветка) |
| `debug/rewrite_onnx_gemm_to_matmul.py` | Переписывает `Gemm` в `MatMul` как workaround. | ✅ Utility (debug-ветка) |
| `debug/rewrite_onnx_extmaps_bias_inputs_to_fp16.py` | Переписывает extmaps/extbias входы в FP16 и убирает лишние Cast. | ✅ Utility (debug-ветка) |
| `debug/qnn_onnx_converter_expanddims_patch.py` | Monkey-patch/entrypoint для QAIRT converter с нужными фикcами. | ⚠️ Низкоуровневый workaround |
| `debug/assess_generated_image.py` | Даёт быструю безреференсную оценку итоговой картинки. | ✅ Полезный utility |

### Experimental / alternative

| Файл | Что делает | Статус |
| --- | --- | --- |
| `debug/test_distillation_loras.py` | Проверяет альтернативные distillation/LoRA сценарии. | ⚠️ Чистое исследование |

## Shell-helpers рядом с Python-цепочкой

Они не Python, но важны для понимания полной картины:

| Файл | Что делает |
| --- | --- |
| `debug/build_fp16_ctx.sh` | Старый rooted helper для генерации context binary на телефоне. |
| `debug/run_ctxgen_lightning.sh` | Более свежий helper для phone-side `qnn-context-binary-generator`. |

## Что реально входит в “финальный путь”, а что нет

### Финальный путь ближе к public beta

- `phone_generate.py`
- `scripts/deploy_to_phone.py`
- tokenizer + context binaries
- APK

### Не финальный путь, а исследовательская лаборатория

Следующие файлы **не нужно воспринимать как обязательные для обычного пользователя**:

- все `debug/compare_*`
- все `debug/generate_*reference*`
- `debug/trace_unet_layer_parity.py`
- `debug/check_encoder_outputs.py`
- `debug/measure_ram.py`
- `debug/test_distillation_loras.py`
- AI Hub-ветка (`debug/export_and_compile_aihub.py`, `debug/export_split_unet.py`)

## Почему их так много — и это не мусор

Этот набор вырос из реальной практики:

- часть файлов появилась из-за QAIRT/QNN edge-case'ов;
- часть — из-за необходимости проверить parity между PyTorch, ONNX и телефоном;
- часть — из-за того, что split-UNet и Lightning/full-UNet ветки развивались параллельно;
- часть — это просто честная R&D-инфраструктура, а не “лишние скрипты”.

Именно поэтому в README теперь важно явно разделять:

- **что рекомендовано новичку**;
- **что нужно автору/исследователю**;
- **что пока экспериментально**.
