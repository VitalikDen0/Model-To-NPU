#!/system/bin/sh
# Generate HTP context binary for Lightning UNet model

BASE=/data/local/tmp/sdxl_qnn
MODEL_LIB=$BASE/model/libunet_lightning8step.so
CONFIG=$BASE/htp_backend_extensions_lightning.json
OUTPUT=$BASE/context/unet_lightning8step.serialized.bin
TIMEOUT=${CTX_TIMEOUT_SEC:-1800}

export LD_LIBRARY_PATH="$BASE/lib:$BASE/bin:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$BASE/lib;/vendor/lib64/rfs/dsp;/vendor/lib/rfsa/adsp;/vendor/dsp"

echo "=== Lightning context binary generation ==="
echo "  MODEL: $MODEL_LIB"
echo "  CONFIG: $CONFIG"
echo "  OUTPUT: $OUTPUT"
echo "  TIMEOUT: ${TIMEOUT}s"

mkdir -p $BASE/context
cd $BASE

timeout $TIMEOUT $BASE/bin/qnn-context-binary-generator \
    --backend libQnnHtp.so \
    --model "$MODEL_LIB" \
    --binary_file "$OUTPUT" \
    --config_file "$CONFIG" \
    --log_level verbose \
    2>&1

EXIT=$?
echo "EXIT:$EXIT"

if [ -f "${OUTPUT}.bin" ]; then
    SIZE=$(stat -c%s "${OUTPUT}.bin" 2>/dev/null || echo "?")
    echo "Context binary created: ${OUTPUT}.bin ($SIZE bytes)"
elif [ -f "$OUTPUT" ]; then
    SIZE=$(stat -c%s "$OUTPUT" 2>/dev/null || echo "?")
    echo "Context binary created: $OUTPUT ($SIZE bytes)"
else
    echo "ERROR: Context binary not created"
fi
