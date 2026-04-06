# SDXL runbook: реально использованные файлы и команды

Актуально для подтверждённого прогона от `2026-04-06`.

## 1) База модели

- Основной checkpoint для старта:
  - `J:\ComfyUI\models\checkpoints\waiIllustriousSDXL_v160.safetensors`
- Теперь `SDXL/run_end_to_end.ps1` и `scripts/build_all.py` умеют **спрашивать путь интерактивно**.
  - если нажать Enter, берётся дефолт: `J:\ComfyUI\models\checkpoints\waiIllustriousSDXL_v160.safetensors`

## 2) Файлы, реально используемые в рабочем (core) пути

### Core-скрипты в `SDXL/`

- `SDXL/run_end_to_end.ps1` — orchestration build/deploy/smoke.
- `SDXL/convert_sdxl_checkpoint_to_diffusers.py` — checkpoint -> diffusers.
- `SDXL/bake_lora_into_unet.py` — merge Lightning LoRA в UNet.
- `SDXL/export_clip_vae_to_onnx.py` — экспорт CLIP-L / CLIP-G / VAE в ONNX.
- `SDXL/export_sdxl_to_onnx.py` — экспорт UNet в ONNX.

### Orchestration/деплой в `scripts/`

- `scripts/build_all.py` — быстрый reproducible helper ранних этапов.
- `scripts/deploy_to_phone.py` — деплой runtime-файлов на телефон.

### Runtime-генерация

- `phone_generate.py` (в репозитории) -> деплоится как `phone_gen/generate.py` на телефоне.

## 3) Файлы на телефоне, которые участвуют в генерации

Базовый runtime path (пример):

- `/data/local/tmp/sdxl_qnn` (rooted layout)
- или `/sdcard/Download/sdxl_qnn` (shared-storage layout)

Обязательные подпапки/файлы:

- `context/clip_l.serialized.bin.bin`
- `context/clip_g.serialized.bin.bin`
- `context/unet_encoder_fp16.serialized.bin.bin`
- `context/unet_decoder_fp16.serialized.bin.bin`
- `context/vae_decoder.serialized.bin.bin`
- `bin/qnn-net-run`
- `lib/libQnnHtp.so` (+ сопутствующие QNN runtime libs)
- `phone_gen/generate.py`
- `phone_gen/tokenizer/vocab.json`
- `phone_gen/tokenizer/merges.txt`

## 4) Команды для реального прогона

## 4.1 Build-only до границы телефона

```powershell
pwsh SDXL/run_end_to_end.ps1 -OutputRoot build/sdxl_work_wai160_20260406 -SkipDeploy -SkipSmokeTest
```

Скрипт сам спросит:

- `Path to SDXL checkpoint (.safetensors) [J:\ComfyUI\models\checkpoints\waiIllustriousSDXL_v160.safetensors]`

## 4.2 Если нужно явно указать checkpoint

```powershell
pwsh SDXL/run_end_to_end.ps1 -Checkpoint "J:\ComfyUI\models\checkpoints\waiIllustriousSDXL_v160.safetensors" -OutputRoot build/sdxl_work_wai160_20260406 -SkipDeploy -SkipSmokeTest
```

## 4.3 Деплой на телефон

```powershell
python scripts/deploy_to_phone.py --contexts-dir <путь_к_context_binaries> --phone-base /data/local/tmp/sdxl_qnn
```

(при необходимости добавляются `--qnn-lib-dir` и `--qnn-bin-dir`)

## 4.4 Нативная генерация на телефоне (Termux Python через root)

```powershell
adb shell "su --mount-master -c 'export PATH=/data/data/com.termux/files/usr/bin:\$PATH; export SDXL_QNN_BASE=/data/local/tmp/sdxl_qnn; /data/data/com.termux/files/usr/bin/python3 /data/local/tmp/sdxl_qnn/phone_gen/generate.py \"orange cat on wooden chair, detailed fur, soft cinematic light, high quality\" --seed 777 --steps 8 --cfg 3.5 --prog-cfg --name wai160_phone_native_cfg35_20260406'"
```

## 4.5 Забрать итоговую картинку на ПК

```powershell
adb pull /data/local/tmp/sdxl_qnn/outputs/wai160_phone_native_cfg35_20260406.png NPU/outputs/wai160_phone_native_cfg35_20260406.png
```

## 5) Где смотреть результат

- итоговый PNG:
  - `NPU/outputs/wai160_phone_native_cfg35_20260406.png`
- build-артефакты:
  - `build/sdxl_work_wai160_20260406/`

## 6) Что считается "core" после cleanup

В корне `SDXL/` оставлены только реально нужные для рабочей цепочки скрипты.

- Все остальные конвертеры, патчи, rewrite-утилиты, TAESD/QNN ветки, host-side debug-генераторы и исследовательские инструменты перенесены в `SDXL/debug/`.
