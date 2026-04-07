# Живой пример SDXL-раскладки на телефоне

Этот файл документирует **реальный снимок структуры каталогов**, собранный **2026-03-29** с подключённого телефона через `adb shell su -c ...`.

Он отражает **историческую rooted-раскладку**. Текущая основная non-root цель деплоя, описанная в репозитории, — `/sdcard/Download/sdxl_qnn`.

Файл включён в репозиторий как пример того, что toolchain уже реально создавал на устройстве.  
Крупные бинарные файлы в git **не** хранятся — это только инвентаризация и справка.

> **Примечание:** текущий live preview runtime теперь в первую очередь предпочитает rebuilt QNN TAESD preview assets на GPU (`context/taesd_decoder.serialized.bin.bin` вместе с `model/libTAESDDecoder.so`) и откатывается на маленький CPU-side ONNX decoder (`phone_gen/taesd_decoder.onnx`) только когда QNN preview недоступен. Этот rooted-снимок всё ещё полезен как исторический пример раскладки.

## Базовый путь

`/data/local/tmp/sdxl_qnn`

## Наблюдавшиеся top-level элементы

- `bin/`
- `context/`
- `lib/`
- `model/`
- `output/`
- `outputs/`
- `phone_gen/`
- `runtime_work_gen/`
- `python3` (symlink на Termux Python)
- helper scripts и QNN config JSON files

## Обнаруженные context binaries

```text
clip_l.serialized.bin.bin
clip_g.serialized.bin.bin
unet_encoder_fp16.serialized.bin.bin
unet_decoder_fp16.serialized.bin.bin
unet_lightning8step.serialized.bin.bin
vae_decoder.serialized.bin.bin
```

## Обнаруженные model libraries

```text
libclip_l.so
libclip_g.so
libunet_lightning_fp16.so
libunet_lightning8step.so
libvae_decoder.so
```

## Обнаруженные файлы phone-side генератора

```text
/data/local/tmp/sdxl_qnn/phone_gen/generate.py
/data/local/tmp/sdxl_qnn/phone_gen/tokenizer/vocab.json
/data/local/tmp/sdxl_qnn/phone_gen/tokenizer/merges.txt
```

## Дополнительные live-артефакты, присутствовавшие на устройстве

В живом rooted-снимке также были файлы, полезные для разработки и отладки, но **не входящие в минимально необходимую раскладку**:

- `ctxgen_fp16.log`
- `ctxgen_fp16_exit.txt`
- `htp_backend_ext_config_*.json`
- `htp_backend_extensions_*.json`
- `runtime_work_gen/`
- helper shell scripts вроде `run_cfg35.sh`, `run_seed15.sh` и `run_test.sh`

## Зачем нужен этот файл

- показать конкретный результат, реально произведённый tooling'ом;
- помочь сравнить документированную минимальную раскладку с живым rooted deployment;
- избежать коммита многогигабайтных бинарников в git.

## Связанные примеры

- rooted sample artifacts: [`rooted-phone-sample/README.md`](rooted-phone-sample/README.md)
- rooted sample artifacts (RU): [`rooted-phone-sample/README_RU.md`](rooted-phone-sample/README_RU.md)
