# Историческая хронология производительности и архив оптимизаций

Этот документ сохраняет исторические замеры производительности, эксперименты по оптимизации и технические заметки с ранних этапов разработки. Записи хранятся как справочный материал и для прозрачности.

**Языки:** [English](HISTORY_EN.md) | [Русский](HISTORY_RU.md)

В текущей истории git ветка `0.3.x` представлена единственным тегом **`v0.3.0`**.

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

### v0.2.4 — Native C ускоритель / groundwork для runtime

- Опциональный native C ускоритель для scheduler/layout hot spots.
- Переходный snapshot; точный APK-артефакт не сохранён.
- В рабочем commit message ещё использовалась метка `v0.2.4-beta`, но опубликованный тег в репозитории — именно **`v0.2.4`**.

### v0.2.5 — Burst mode, фикс staging runtime accel

- QNN `burst` по умолчанию.
- Фикс staging native C accelerator для Android shared-storage `dlopen`.
- Локальный review: **75.6 с итого** (`CLIP 2.774 s`, `UNet 66.639 s`, `VAE 2.960 s`).

---

## Архив релизной линии 0.4.x

Линия `0.4.x` — это период, когда репозиторий ушёл от одного фиксированного SDXL-пути к более практичной phone-side продуктовой линии с APK-упаковкой, доставкой runtime, экспериментами с shared reuse и последующим расследованием decoder-regression.

### v0.4.0 — Переменное разрешение + foundations для self-contained APK

- `phone_generate.py` перестал считать, что всё всегда равно `1024×1024`;
- появились resolution-scoped каталоги контекстов через `context/{W}x{H}/`;
- `export_split_unet.py` и `build_all.py` получили resolution-aware export/build path;
- APK получил управление шириной/высотой и начал экспортировать `SDXL_QNN_WIDTH` / `SDXL_QNN_HEIGHT`;
- был заложен фундамент для более автономного runtime/package пути.

### v0.4.1 — Bundled runtime payload и фикс stale-script drift

- APK начал включать в себя актуальный phone runtime payload (`generate.py`, server binary, optional accel);
- app-side execution стал предпочитать bundled runtime вместо устаревших файлов на телефоне;
- runtime verify/inspect path стал лучше диагностировать рассинхрон аргументов/argparse;
- deploy flow стал надёжнее пушить `qnn-multi-context-server` и resolution-scoped contexts.

### v0.4.2 — Нет чисто оформленного публичного milestone

- В текущей git-истории не найден чёткий публичный `v0.4.2` как отдельно оформленный стабильный релизный этап.
- По факту линия быстро перетекла в серию фиксов и доработок, которые уже оформлялись как `v0.4.3`.

### v0.4.3 — Серия фиксов shared prewarm reuse и runtime hotfixes

- app-open prewarm и foreground generation переделывались в сторону shared reuse вместо одноразовых helper-процессов;
- bundled runtime delivery стала более детерминированной, включая refresh логики payload;
- staging TAESD payload и preview/runtime path были подтянуты;
- отдельно лечились shared-server startup и FIFO readiness проблемы;
- по сути это не одна правка, а серия коммитов (`010331d`, `a3e9376`, `e987a11`, `c5bac22`).

### v0.4.4 — Preset-only smoothing для APK

- APK убрал произвольное ручное `WxH` редактирование в пользу проверенных preset-ов;
- preview/final decode-display path был сглажен, чтобы меньше давить на UI;
- APK-side QNN profile стал мягче, чтобы уменьшить whole-device lag / crash risk.

### v0.4.5 — Stability rollback после проблемного shared reuse

- foreground generation ушёл от более агрессивного shared prewarm/server reuse пути;
- линия APK откатилась к более безопасному поведению после риска late-step freeze;
- foreground path вернулся к `burst` профилю.

### v0.4.6 — Stability-first refresh для APK

- в публичной линии background prewarm остался отключённым;
- детерминированный runtime payload refresh был сохранён;
- runtime staging и packaging стали более консервативными и воспроизводимыми;
- эта линия уже документировалась вокруг знакомого маркера **34.6 с cold-start APK**.

### v0.4.7 — Hotfix по CFG / TAESD UX

- точные пользовательские CFG значения, включая `1.0`, начали прокидываться корректно;
- TAESD/live-preview failures стали показываться как нефатальные предупреждения вместо молчаливой путаницы;
- docs и proof references были приведены к актуальному публичному состоянию.

### v0.4.8-beta — Bundled Python runtime, dual paths, TAESD off в APK

- APK получил bundled Python 3.13 runtime, чтобы меньше зависеть от внешнего Termux-пути;
- появился dual base-dir handling для root/no-root сценариев;
- TAESD preview был намеренно отключён в APK, потому что shared HTP-preview вредил fast path, а GPU backend loading внутри app-process ещё был проблемным.

### v0.4.8-beta2 — Runtime bug fix и улучшение error UX

- в Android-линии были исправлены runtime bug'и;
- улучшен explicit error-state handling, включая отдельную кнопку копирования ошибки;
- safer стали seed parsing и соседние input-path сценарии.

### v0.4.8-beta3 — Фикс скорости decoder и честная фиксация остаточного хвоста

- для `qnn-multi-context-server` был усилен HTP perf mode (`DCVS` off, MAX corners, RPC latency/polling controls);
- локально decoder latency ушёл из класса `~820 мс` в диапазон примерно `~725–776 мс`;
- остаточный хвост порядка ~`50 мс` относительно исторического ideal marker остался и был зафиксирован явно, а не спрятан.

### v0.5.0 — Разделение SDXL / WAN и обновление интерфейса APK

- Разделение визуальных вкладок SDXL и WAN 2.1 с независимым управлением параметрами;
- Внедрение динамического сканирования папок `context/` и `context/lora_slots/` для автоматической подгрузки слотов на лету;
- Оптимизация LMK (Low Memory Killer): добавлены кнопки принудительной выгрузки памяти NPU и автоматическая очистка при выходе/смене моделей.

### v0.5.1 — Релиз подготовки к динамической хирургии графов NPU

- Обновление Android APK (`v0.5.1`), полная защита от LMK и интеграция dynamic directory scanners;
- Фикс конвертера QNN на Windows (`sanitize_onnx_types.py`): приведение типов INT64 к INT32 предотвращает выравнивающие сбои и панику StridedSlice;
- Публикация нового технического роадмапа (`ROADMAP_DYNAMIC_NPU_SURGERY`): пересмотр парадигмы статических бакетов разрешений и смены LoRA в пользу динамической инъекции весов и маскирования внимания (`-10000.0` вместо $-\infty$ во избежание сжатия диапазона квантования);
- Ревизия архитектурных экспериментов: сформулировано решение для истинного Zero-Copy через единый зарегистрированный блок `rpcmem`.

---

## Архив экспериментов по оптимизации

### Zero-copy pointer swap (Ограничение QNN HTP & Решение)

- *Первоначальная попытка:* Перестановка указателей буферов decoder input на encoder output для устранения `memcpy` в `RUN_CHAIN` вызывала **QNN error 6004** из-за незарегистрированных `Qnn_MemHandle`.
- *Архитектурное решение:* Вместо подмены указателей на лету при старте `qnn-multi-context-server` выделяется единый общий блок `rpcmem`. При инициализации NPU Encoder этот блок регистрируется как Output Buffer, а при инициализации NPU Decoder — как Input Buffer. Это исключает вызов `memcpy` и экономит ~80 МБ передачи данных на каждом шаге.

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
