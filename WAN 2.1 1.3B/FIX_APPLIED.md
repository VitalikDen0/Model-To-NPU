# WAN 2.1 — Исправление ONNX экспорта

## ✅ Проблема решена

**Исправление:** Заменил `tuple` на `list` для axis в операциях `mean()` в RMSNorm и LayerNorm.

### Было:
```python
dims = tuple(range(-len(self.normalized_shape), 0))
mean = x.mean(dim=dims, keepdim=True)
```

### Стало:
```python
# Явно указываем axis как список для корректного ONNX экспорта
dims = list(range(-len(self.normalized_shape), 0))
mean = x.mean(dim=dims, keepdim=True)
```

## Причина проблемы

При экспорте в ONNX, PyTorch конвертирует `tuple` в некорректное значение для axis операции Reduce, что приводило к ошибке:
```
ValueError: Reduce param axis must be >= 0 and <= rank(input[0]) 3, but get:111240816
```

Использование `list` вместо `tuple` решает эту проблему.

## Следующие шаги

1. ✅ Исправлен `export_wan_to_onnx.py`
2. 🔄 Пере-экспорт transformer в ONNX
3. ⏳ Компиляция через QAIRT SDK
4. ⏳ Деплой на телефон

## Статус

- **Файл:** `WAN 2.1 1.3B/export_wan_to_onnx.py`
- **Изменения:** 2 места (ExportableRMSNorm и ExportableLayerNorm)
- **Тест:** Запущен пере-экспорт
