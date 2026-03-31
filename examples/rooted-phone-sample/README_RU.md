# Rooted SDXL phone sample (набор референсных артефактов)

В этой папке лежит **небольшой реальный набор файлов**, скопированный с rooted-развёртывания телефона из:

`/data/local/tmp/sdxl_qnn`

Папка нужна как **референс, демонстрация и учебный пример**, чтобы можно было потрогать рабочую раскладку даже если полный пайплайн у вас пока ещё не собирается от начала до конца.

## Что сюда включено

Здесь специально оставлены только **лёгкие и показательные** файлы:

- несколько реально сгенерированных PNG-картинок;
- текущая маленькая TAESD preview-модель: `phone_gen/taesd_decoder.onnx`;
- HTP backend JSON-конфиги;
- helper shell scripts из rooted-экспериментов;
- `phone_gen/generate.py` из реального phone-side runtime;
- файлы `phone_gen/tokenizer/` в том виде, в каком они были на устройстве;
- небольшие логи вроде `ctxgen_fp16.log`.

## Что намеренно НЕ включено

Эта папка **не** раздаёт тяжёлую или чувствительную часть runtime-набора:

- основные context binaries из `context/` (`clip_*`, `unet_*`, `vae_*`);
- QNN runtime libraries из `lib/`;
- скомпилированные model libraries из `model/`;
- полные проприетарные SDK-наборы.

Причины простые: они слишком большие, завязаны на конкретную среду или просто не подходят для такого распространения.

Почему их нет: обычный GitHub-репозиторий не принимает multi-hundred-megabyte / multi-gigabyte model payload'ы. Одни только split UNet/CLIP context'ы заметно превышают лимит на файл, поэтому в sample-наборе лежат только лёгкие куски, которые реально имеет смысл версионировать.

Текущее «лёгкое исключение» для live preview — маленький TAESD ONNX decoder: его удобно хранить в репозитории, и он остаётся CPU fallback-путём. Но сам живой runtime теперь в первую очередь предпочитает rebuilt QNN TAESD preview assets на GPU (`taesd_decoder.serialized.bin.bin` вместе с `libTAESDDecoder.so`) и откатывается на ONNX только если QNN preview assets отсутствуют или ломаются. В старых rooted-снимках всё ещё могут попадаться legacy TAESD context-файлы.

## Ссылка на исходную модель

Этот rooted sample был получен в экспериментах вокруг следующего публичного сочетания:

- базовый checkpoint: [WAI Illustrious SDXL v1.60](https://civitai.com/models/827184/wai-illustrious-sdxl?modelVersionId=2514310)
- LoRA: [SDXL-Lightning 8-step LoRA](https://huggingface.co/ByteDance/SDXL-Lightning)

## Права и дисклеймер

Этот репозиторий **не претендует** на права на исходные model weights, checkpoint'ы или их идентичность.

- все права на исходную модель принадлежат её автору(ам) и/или издателю(ям);
- все права на SDXL-Lightning принадлежат её автору(ам) и/или издателю(ям);
- содержимое этой папки публикуется строго как **справочный/учебный пример** rooted phone-side раскладки и вспомогательных артефактов;
- если вам нужна сама исходная модель, берите её из официального upstream-источника.

## Связанные файлы

- английское описание rooted-layout: [`../phone-sdxl-qnn-layout.md`](../phone-sdxl-qnn-layout.md)
- русское описание rooted-layout: [`../phone-sdxl-qnn-layout_RU.md`](../phone-sdxl-qnn-layout_RU.md)
- английская версия описания этой папки: [`README.md`](README.md)
