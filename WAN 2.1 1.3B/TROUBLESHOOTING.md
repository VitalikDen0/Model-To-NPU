# WAN 2.1 — Проблемы и решения

## Текущая проблема с AI Hub

**Job ID:** `j5q7nw7og`
**Статус:** FAILED
**Ошибка:** "Conversion to context binary failed with exit code 15"

### Анализ

1. **Размер модели:** 8.3 GB external data — очень большая модель
2. **ONNX opset:** 18 (поддерживается QNN)
3. **Операции:** Стандартные (Add, MatMul, Softmax, etc.)
4. **Shapes:** Все статические, нет dynamic shapes

### Возможные причины exit code 15

1. **Timeout на AI Hub** — модель слишком большая для конвертации в облаке
2. **Memory limit** — QNN конвертер не может обработать 8.3 GB модель
3. **Unsupported pattern** — какой-то специфический паттерн операций не поддерживается

### Решения

#### Вариант 1: Локальная компиляция (рекомендуется)
Компилировать QNN context binary локально на хосте с QAIRT SDK:
- Больше контроля
- Можно видеть детальные логи
- Нет ограничений по времени/памяти

#### Вариант 2: Упрощение модели
- Квантизация до INT8 перед экспортом
- Разделение на более мелкие части
- Уменьшение sequence length (128 → 64)

#### Вариант 3: Другие параметры AI Hub
```python
compile_options = (
    "--target_runtime qnn_context_binary "
    "--qnn_options default_graph_htp_precision=FLOAT16 "
    "--qnn_options htp_socs=sm8750 "  # Явно указать SoC
    "--qnn_options enable_htp_weight_sharing=1"  # Оптимизация памяти
)
```

## Следующие шаги

1. Попробовать локальную компиляцию с QAIRT SDK
2. Если не получится — упростить модель
3. Параллельно начать работу с Flux.1 и SD3.5 (они меньше)

## Статус других моделей

### Flux.1 Dev
- Размер: ~12 GB
- Статус: Готов к скачиванию
- Скрипт: `Flux.2/download_flux.py`

### SD3.5 Large Turbo
- Размер: ~11.9 GB
- Статус: Готов к скачиванию
- Скрипт: `SD3.5/download_sd35.py`
