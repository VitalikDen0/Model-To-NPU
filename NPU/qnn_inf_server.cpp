/**
 * qnn_inf_server.cpp  —  Persistent QNN Inference Server for Hexagon NPU
 *
 * WHY THIS EXISTS
 * ───────────────
 * qnn-net-run reloads the context binary on every invocation:
 *   1. CDSP runtime init & DSP firmware context setup   ~1-3 s
 *   2. Context binary file read + graph parsing          ~1-2 s
 *   3. Input/output buffer allocation                   ~0.1-0.5 s
 *   4. Actual NPU inference (FP16 UNet half)             ~0.05-0.3 s  ← actual work!
 *
 * For an 8-step generation with 2 subprocess calls/step = 16 calls:
 *   Current:  7 s/call × 16 calls = 112 s   (only ~48 ms is real compute!)
 *   Daemon:   7 s (first call) + ~0.3 s × 15 = ~11.5 s   (≈10× speedup)
 *
 * PROTOCOL
 * ────────
 * Daemon reads commands from a FIFO (named pipe). Per inference:
 *   1. Client writes input_list path  (newline-terminated)
 *   2. Client writes output_dir path  (newline-terminated)
 *   3. Daemon runs inference, writes to output_dir
 *   4. Daemon writes "OK\n" or "ERR:<msg>\n" to response FIFO
 *
 * FIFOs:
 *   /data/local/tmp/sdxl_qnn/daemon_enc.req   (request, context = encoder)
 *   /data/local/tmp/sdxl_qnn/daemon_enc.rsp   (response)
 *   /data/local/tmp/sdxl_qnn/daemon_dec.req   (request, context = decoder)
 *   /data/local/tmp/sdxl_qnn/daemon_dec.rsp   (response)
 *
 * BUILD
 * ─────
 * Requires: QNN SDK 2.31+ headers, Android NDK 28+
 *
 *   NDK=C:/Users/vital/AppData/Local/Android/Sdk/ndk/28.2.13676358
 *   QNN=C:/Qualcomm/AIStack/QAIRT/2.31.0.250130
 *   CC=$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin/aarch64-linux-android35-clang++
 *
 *   $CC -std=c++17 -O2 \
 *     -I$QNN/include/QNN \
 *     -L$QNN/lib/aarch64-android \
 *     -Wl,-rpath,'$ORIGIN/../lib' \
 *     qnn_inf_server.cpp \
 *     -lQnnHtp -lQnnSystem \
 *     -o qnn_inf_server_android
 *
 *   adb -s e01ad23a push qnn_inf_server_android /data/local/tmp/sdxl_qnn/bin/
 *   adb -s e01ad23a shell chmod 755 /data/local/tmp/sdxl_qnn/bin/qnn_inf_server_android
 *
 * PHONE-SIDE LAUNCH (start before generation)
 * ─────────────────
 *   DR=/data/local/tmp/sdxl_qnn
 *   # Create FIFOs once
 *   mkfifo $DR/daemon_enc.req $DR/daemon_enc.rsp
 *   mkfifo $DR/daemon_dec.req $DR/daemon_dec.rsp
 *   # Start servers (each loads ONE context and loops forever)
 *   LD_LIBRARY_PATH=$DR/lib ADSP_LIBRARY_PATH="$DR/lib;/vendor/lib64/rfs/dsp" \
 *     $DR/bin/qnn_inf_server_android \
 *       $DR/context/unet_encoder_fp16.serialized.bin.bin \
 *       $DR/daemon_enc.req $DR/daemon_enc.rsp &
 *   LD_LIBRARY_PATH=$DR/lib ADSP_LIBRARY_PATH="$DR/lib;/vendor/lib64/rfs/dsp" \
 *     $DR/bin/qnn_inf_server_android \
 *       $DR/context/unet_decoder_fp16.serialized.bin.bin \
 *       $DR/daemon_dec.req $DR/daemon_dec.rsp &
 *
 * TODO: integrate phone_generate.py → use daemon FIFOs instead of qnn_run() subprocess.
 */

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// QNN Runtime C API headers (from QAIRT SDK)
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpDevice.h"
#include "System/QnnSystemInterface.h"

// ─── Minimal QNN context-binary loader ───────────────────────────────────────
// Full implementation needs Qnn_LogHandle_t, Qnn_BackendHandle_t, etc.
// This is a skeleton — fill in from QNN SDK samples/QnnSampleApp.

struct InferenceServer {
    // handles
    Qnn_LogHandle_t     log_handle    = nullptr;
    Qnn_BackendHandle_t backend       = nullptr;
    Qnn_DeviceHandle_t  device        = nullptr;
    Qnn_ContextHandle_t context       = nullptr;
    Qnn_GraphHandle_t   graph         = nullptr;

    // For retrieve_context path we also need:
    QnnSystemContext_Handle_t sys_ctx  = nullptr;

    // Input/output tensor metadata (filled after context load)
    std::vector<Qnn_Tensor_t> inputs;
    std::vector<Qnn_Tensor_t> outputs;

    bool load_context(const char* ctx_path,
                      const char* backend_lib,    // e.g. libQnnHtp.so
                      const char* system_lib) {   // e.g. libQnnSystem.so
        // TODO: dlopen backend_lib, get QNN interface
        // TODO: QnnLog_create(...)
        // TODO: QnnBackend_create(...)
        // TODO: QnnDevice_create(...)
        // TODO: QnnSystemContext_create(...)
        // TODO: QnnSystemContext_getBinaryInfo(ctx_path, ...) → retrieve graph metadata
        // TODO: QnnContext_createFromBinaryFile(ctx_path, ...) → loads into CDSP
        // TODO: QnnGraph_retrieve(context, graph_name, &graph)
        // TODO: QnnGraph_getGraphInfo(...) → populate inputs/outputs
        fprintf(stderr, "[daemon] TODO: implement QNN context loading\n");
        fprintf(stderr, "[daemon] See QNN SDK samples: samples/QNN/SampleApp/\n");
        return false;
    }

    bool run_inference(const char* input_list_path, const char* output_dir) {
        // Parse input_list (space-separated paths per line, one line per inference batch)
        // For each line:
        //   1. Read raw files into Qnn_Tensor_t memHandles
        //   2. QnnGraph_execute(graph, inputs.data(), N, outputs.data(), M, nullptr, nullptr)
        //   3. Write output tensors to output_dir/Result_0/output_N.raw
        fprintf(stderr, "[daemon] TODO: implement inference loop\n");
        return false;
    }

    void free_context() {
        if (context) { /* QnnContext_free(context, nullptr); */ }
        if (device)  { /* QnnDevice_free(device, nullptr); */ }
        if (backend) { /* QnnBackend_free(backend, nullptr); */ }
        if (log_handle) { /* QnnLog_free(log_handle); */ }
    }
};

// ─── FIFO protocol loop ──────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <context.bin> <req.fifo> <rsp.fifo>\n"
            "       [--backend libQnnHtp.so] [--system libQnnSystem.so]\n",
            argv[0]);
        return 1;
    }

    const char* ctx_path  = argv[1];
    const char* req_fifo  = argv[2];
    const char* rsp_fifo  = argv[3];

    // Optional flags
    const char* backend_lib = "libQnnHtp.so";
    const char* system_lib  = "libQnnSystem.so";
    for (int i = 4; i < argc - 1; ++i) {
        if (!strcmp(argv[i], "--backend")) backend_lib = argv[i+1];
        if (!strcmp(argv[i], "--system"))  system_lib  = argv[i+1];
    }

    fprintf(stderr, "[daemon] Loading context: %s\n", ctx_path);
    InferenceServer server;

    if (!server.load_context(ctx_path, backend_lib, system_lib)) {
        fprintf(stderr, "[daemon] FATAL: context load failed\n");
        return 1;
    }
    fprintf(stderr, "[daemon] Context loaded. Listening on %s\n", req_fifo);

    // Inference loop: read request, run, respond
    while (true) {
        // Open request FIFO (blocking open — waits for client write)
        FILE* req = fopen(req_fifo, "r");
        if (!req) { perror("fopen req"); break; }

        char input_list_path[4096] = {};
        char output_dir[4096]      = {};
        if (!fgets(input_list_path, sizeof(input_list_path), req)) { fclose(req); break; }
        if (!fgets(output_dir,      sizeof(output_dir),      req)) { fclose(req); break; }
        fclose(req);

        // Strip newlines
        for (char* p : {input_list_path, output_dir})
            for (char* c = p; *c; ++c) if (*c == '\n' || *c == '\r') *c = '\0';

        fprintf(stderr, "[daemon] Running: %s -> %s\n", input_list_path, output_dir);

        const bool ok = server.run_inference(input_list_path, output_dir);

        // Write response
        FILE* rsp = fopen(rsp_fifo, "w");
        if (!rsp) { perror("fopen rsp"); break; }
        fprintf(rsp, "%s\n", ok ? "OK" : "ERR:inference failed");
        fclose(rsp);
    }

    server.free_context();
    return 0;
}
