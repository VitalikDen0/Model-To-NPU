# Историческая хронология производительности и архив оптимизаций

Этот документ сохраняет исторические замеры производительности, эксперименты по оптимизации и технические заметки с ранних этапов разработки. Записи хранятся как справочный материал и для прозрачности.

**Языки:** [English](HISTORY_EN.md) | [Русский](HISTORY_RU.md)

---

## Историческая хронология

### v0.1.2 — Live Preview (TAESD)

- APK получил опциональный **Live Preview (TAESD)** через `phone_gen/taesd_decoder.onnx` + `onnxruntime` прямо на телефоне.

### v0.1.3 — QNN mmap, первые оптимизации (2026-03-31)

- Phone runtime и APK-путь включают QNN `mmap` по умолчанию.
- Контрольный прогон на OnePlus 13: **104.4 с итого** (`CLIP 1.993 s`, `UNet 91.466 s`, `VAE 8.992 s`) при `1024×1024`, `8` шагов, `CFG=1.0`.

### v0.2.0 — Мониторинг температуры, sustained_high_performance

- Phone runtime и APK показывают живые **CPU / GPU / NPU** температуры.
- Профиль по умолчанию: `sustained_high_performance`.
- Автоматическое подключение HTP backend extensions при наличии `libQnnHtpNetRunExtensions.so`.
- Лучший прогон: **79.7–80.6 с итого** с progressive CFG на OnePlus 13.

### v0.2.1 — App-private cache

- APK перенаправляет временные runtime-файлы через app-private cache вместо shared storage.

### v0.2.2 — Починка TAESD preview

- Починен TAESD preview для QNN-пути.
- APK-парсинг preview-таймингов снова обрабатывает `QNN GPU` preview строки.

### v0.2.3 — Исторический быстрый путь (до reset)

- Split-UNet reuse pass заставил ранние guided шаги постепенно ускоряться вместо зависания у плато ~12 с.
- Один runtime-only прогон достиг **62.0 с итого** (`CLIP 1.787 s`, `UNet 55.980 s`, `VAE 3.138 s`) с выключенным Live Preview.
- Этот прогон был реальным, но точное состояние телефона не было заархивировано до factory reset, поэтому теперь он является историческим и не воспроизводимым.
- Прогрессия UNet по шагам: CFG шаги 1..4: `9.765 → 8.230 → 8.386 → 7.936 s`; no-guidance шаги 5..8: `5.377 → 5.513 → 5.294 → 5.479 s`.

### v0.2.4-beta — Native C ускоритель

- Опциональный native C ускоритель для scheduler/layout hot spots.
- Переходный snapshot; точный APK-артефакт не сохранён.

### v0.2.5 — Burst mode, фикс staging runtime accel

- QNN `burst` по умолчанию.
- Фикс staging native C accelerator для Android shared-storage `dlopen`.
- Локальный review: **75.6 с итого** (`CLIP 2.774 s`, `UNet 66.639 s`, `VAE 2.960 s`).

---

## Архив экспериментов по оптимизации

### Zero-copy pointer swap (НЕУДАЧА)

Попытка переставить указатели буферов decoder input на encoder output, чтобы убрать memcpy в пайплайне RUN_CHAIN. **QNN error 6004** — QNN HTP использует зарегистрированную shared memory (`rpcmem`). У тензорных буферов есть конкретные memory handles (`Qnn_MemHandle`). Подмена указателей на другие адреса вызывает "Failed to find memHandle", потому что новые адреса не зарегистрированы. **Вывод:** memcpy обязателен для передачи данных между encoder output и decoder input через QNN HTP.

### Persistent daemon подход (РЕГРЕССИЯ)

Использование `qnn-context-runner` как persistent daemon для переиспользования контекстов изначально казалось перспективным, но стабильно давало регрессию на пересобранном телефоне:

- Daemon-all: ~111.3 с → оптимизировано до ~63.3 с (всё ещё медленнее stock ~60.1 с).
- Dummy warmup pass во время CLIP: ~110.5 с (слишком дорого, чтобы скрыть).
- `QnnGraph_setConfig` для VTCM/HVX: ~120.2 с (дальнейшая регрессия).

### Монолитный INT8 UNet (КАТАСТРОФИЧЕСКИ МЕДЛЕННО)

Истинный 8W8A квантованный монолитный UNet из QAIRT 2.44 с anime-calibration:

- Точность: cosine ~0.99913 vs W8A16 контроль (хорошо).
- Скорость: ~161-218 с/шаг vs ~2.55 с/шаг для W8A16 (**63× медленнее**).
- Профайлер подтвердил выполнение на HTP (не CPU fallback), но граф скомпилирован в катастрофически дорогую форму: ~1.35×10¹² accelerator cycles vs ~3.73×10⁹ для W8A16.

### HVX thread ceiling

Backend extension config чувствителен к именам графов. С правильными именами и `hvx_threads=8`, профиль ограничивает до `6`. Потолок в 6 потоков не объясняется термальным тротлингом (cooling device `cdsp_sw_hvx` показывает `cur_state=0`).

### tmpfs workdir (БЕЗ УЛУЧШЕНИЯ)

Перенос `SDXL_QNN_WORK_DIR` в `/tmp` tmpfs не помог и фактически дал регрессию до ~69.4 с (vs ~62.0 с baseline). Остаточный overhead не объясняется одним ext4 workdir I/O.

### Batched CLIP (НЕОДНОЗНАЧНО)

Экспериментальный batched CLIP путь улучшил CLIP время до ~1.83-2.03 с, но ухудшил end-to-end прогоны до ~69.6-70.4 с. Оставлен как opt-in (`SDXL_QNN_BATCH_CLIP=1`).

---

## Подтверждённый полный цикл (2026-04-06)

Checkpoint: `waiIllustriousSDXL_v160.safetensors` (WAI Illustrious SDXL v1.60 + SDXL-Lightning 8-step LoRA).

Артефакты хоста:

- `build/sdxl_work_wai160_20260406/diffusers_pipeline/`
- `build/sdxl_work_wai160_20260406/unet_lightning_merged/`
- `build/sdxl_work_wai160_20260406/onnx_clip_vae/`
- `build/sdxl_work_wai160_20260406/onnx_unet/unet.onnx` + `unet.onnx.data`

Подтверждённый результат: `NPU/outputs/wai160_phone_native_cfg35_20260406.png`

---

## Температурные наблюдения

В прогретых полных прогонах практическая термокартина:

- **CPU:** ~59–70°C
- **GPU:** ~50–52°C
- **NPU:** ~57–72°C (кратковременные пики до ~78°C)
- Первый CPU-пик до `88.8°C` перед первым запуском — скорее всего, переходный скачок сенсора.

---

## Технические заметки

- TAESD preview root cause (2026-04-01): Старый `libTAESDDecoder.so` выдавал значения, обрезанные до `[0,1]` с корреляцией лишь ~0.21 с ONNX. Пересборка из текущего ONNX восстановила диапазон до `[-1.18, 1.23]` и корреляцию ~0.9999.
- После перехода phone runtime на QAIRT 2.44 preview всё ещё был сломан из-за устаревших GPU libs/context от 2.31. Нужна была пересборка и GPU runner, и TAESD context.
- `phone_generate.py::_resolve_exec_binary()` должен создать `WORK_DIR/bin` до staging `qnn-net-run`.
- QAIRT packaging: `libQnnHtpV79Skel.so` может отсутствовать в `lib/aarch64-android` и лежать в `lib/hexagon-v79/unsigned`.
