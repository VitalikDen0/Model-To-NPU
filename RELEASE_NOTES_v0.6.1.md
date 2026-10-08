# Релиз v0.6.1 — Hotfix FastRPC cDSP Transport & Root Detection

Хотфикс для нативного C-движка инференса SDXL на чипах Qualcomm Snapdragon (SM8750 / Hexagon V79).

---

## 🛠️ Исправления и улучшения

### 1. Доступ к Qualcomm FastRPC cDSP (`/dev/fastrpc-cdsp`)
- **Проблема**: В чистом Android Userspace стандартные политики SELinux (`untrusted_app`) блокируют прямое открытие дескриптора `/dev/fastrpc-cdsp` и выделение памяти через ION/DMA-BUF. Это приводило к ошибке FastRPC `4000` (`loadRemoteSymbols failed with err 4000`, `Failed to create transport for device`) и сбою создания контекста QNN `14001`.
- **Решение**: Добавлено автоматическое определение прав суперпользователя (`su`, Magisk, KernelSU, APatch). При наличии root на устройстве движок запускается с привилегиями ядра (`su --mount-master`), выставляя необходимые права (`chmod 666 /dev/fastrpc-cdsp`, `chmod 666 /dev/ion`) и получая беспрепятственный доступ к аппаратным тензорным ядрам Hexagon NPU.

### 2. Мульти-поиск библиотек и DSP Skeleton (`libQnnHtpV79Skel.so`)
- Расширены пути поиска `ADSP_LIBRARY_PATH` и `LD_LIBRARY_PATH`. Теперь драйвер cDSP корректно находит подписанный Hexagon Skel как во встроенном каталоге приложения, так и в `/sdcard/Download/sdxl_qnn/lib`, `/data/local/tmp/sdxl_test/lib` и вендорных путях `/vendor/dsp/cdsp`.
- В нативном C-сервере добавлены явные пути к `/vendor/lib64/libcdsprpc.so` для инициализации `rpcmem` ION.

### 3. Автоматическое обновление рантайма
- Версия бандла обновлена до `native-bundle-v0.6.1`. При обновлении APK приложение автоматически перезаписывает распакованные библиотеки и бинарники в актуальное состояние.
