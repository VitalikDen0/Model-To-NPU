#!/system/bin/sh
cd /data/local/tmp/sdxl_qnn
export LD_LIBRARY_PATH=/data/local/tmp/sdxl_qnn/lib:/data/local/tmp/sdxl_qnn/bin
export ADSP_LIBRARY_PATH="/data/local/tmp/sdxl_qnn/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp"

echo "Starting FP16 UNet context build..."
/data/local/tmp/sdxl_qnn/bin/qnn-context-binary-generator \
  --backend /data/local/tmp/sdxl_qnn/lib/libQnnHtp.so \
  --model /data/local/tmp/sdxl_qnn/model/libunet_lightning_fp16.so \
  --binary_file unet_lightning_fp16.serialized.bin \
  --output_dir /data/local/tmp/sdxl_qnn/context/ \
  --config_file /data/local/tmp/sdxl_qnn/htp_backend_extensions_lightning.json \
  --log_level warn 2>&1

echo "EXIT:$?"
