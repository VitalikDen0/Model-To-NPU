/*
 * qnn_multi_context_server.c - Persistent multi-context QNN server
 *
 * Standalone C program using raw QNN C API (no SampleApp dependency).
 * Loads multiple context binaries, keeps them alive, executes graphs
 * on stdin commands. Eliminates per-inference process spawn + context
 * deserialization overhead.
 *
 * Protocol (stdin/stdout, line-based):
 *   LOAD <id> <context_binary_path>
 *     -> OK <graph_name> <num_inputs> <num_outputs>
 *     -> ERR <message>
 *
 *   RUN <id> <input_list_path> <output_dir>
 *     -> OK <execute_ms>
 *     -> ERR <message>
 *
 *   QUIT
 *     -> OK
 *
 * Build: NDK clang for aarch64-linux-android, link -ldl
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>
#include <ctype.h>
#include <zlib.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

/* QNN headers */
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnCommon.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnDevice.h"
#include "QnnGraph.h"
#include "QnnLog.h"
#include "QnnMem.h"
#include "QnnProfile.h"
#include "QnnProperty.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "HTP/QnnHtpProfile.h"

/* ========================================================================= */
/*  rpcmem for shared DSP memory                                            */
/* ========================================================================= */

#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS  1

typedef void* (*rpcmem_alloc_fn_t)(int heapid, uint32_t flags, int size);
typedef void  (*rpcmem_free_fn_t)(void* po);
typedef int   (*rpcmem_to_fd_fn_t)(void* po);
typedef void  (*rpcmem_init_fn_t)(void);
typedef void  (*rpcmem_deinit_fn_t)(void);

static const char* g_rpc_lib_path = NULL;
static void* g_rpcmem_lib = NULL;
static rpcmem_alloc_fn_t  g_rpcmem_alloc  = NULL;
static rpcmem_free_fn_t   g_rpcmem_free   = NULL;
static rpcmem_to_fd_fn_t  g_rpcmem_to_fd  = NULL;
static rpcmem_init_fn_t   g_rpcmem_init   = NULL;
static rpcmem_deinit_fn_t g_rpcmem_deinit = NULL;
static int g_rpcmem_available = 0;

static void setup_dsp_environment(const char* backend_path, const char* base_dir) {
    char adsp_buf[4096];
    char ld_buf[4096];
    char backend_dir[512] = "";
    if (backend_path) {
        const char* slash = strrchr(backend_path, '/');
        if (slash) {
            size_t len = (size_t)(slash - backend_path);
            if (len < sizeof(backend_dir) - 1) {
                memcpy(backend_dir, backend_path, len);
                backend_dir[len] = '\0';
            }
        }
    }

    /* ADSP_LIBRARY_PATH: semicolon-separated paths used by FastRPC / cDSP loader */
    snprintf(adsp_buf, sizeof(adsp_buf),
        "%s;%s/lib;/data/local/tmp/sdxl_app_engine/lib;/data/user/0/com.sdxlnpu.app/files/termux_bundle/runtime_payload/lib;/data/data/com.sdxlnpu.app/files/termux_bundle/runtime_payload/lib;/data/local/tmp/sdxl_test/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp",
        backend_dir[0] ? backend_dir : "/sdcard/Download/sdxl_qnn/lib",
        base_dir ? base_dir : "/sdcard/Download/sdxl_qnn");

    const char* prev_adsp = getenv("ADSP_LIBRARY_PATH");
    if (prev_adsp && prev_adsp[0]) {
        size_t cur_len = strlen(adsp_buf);
        snprintf(adsp_buf + cur_len, sizeof(adsp_buf) - cur_len, ";%s", prev_adsp);
    }
    setenv("ADSP_LIBRARY_PATH", adsp_buf, 1);
    fprintf(stderr, "[server] ADSP_LIBRARY_PATH: %s\n", adsp_buf);

    /* LD_LIBRARY_PATH: colon-separated paths used by bionic dynamic linker */
    snprintf(ld_buf, sizeof(ld_buf),
        "%s:%s/lib:/data/local/tmp/sdxl_app_engine/lib:/data/user/0/com.sdxlnpu.app/files/termux_bundle/runtime_payload/lib:/vendor/lib64:/system/lib64",
        backend_dir[0] ? backend_dir : "/sdcard/Download/sdxl_qnn/lib",
        base_dir ? base_dir : "/sdcard/Download/sdxl_qnn");
    const char* prev_ld = getenv("LD_LIBRARY_PATH");
    if (prev_ld && prev_ld[0]) {
        size_t cur_len = strlen(ld_buf);
        snprintf(ld_buf + cur_len, sizeof(ld_buf) - cur_len, ":%s", prev_ld);
    }
    setenv("LD_LIBRARY_PATH", ld_buf, 1);
}

static void init_rpcmem(const char* backend_path, const char* base_dir) {
    char path[1024];

    /* 1. Explicit --rpc_lib argument */
    if (g_rpc_lib_path && g_rpc_lib_path[0]) {
        g_rpcmem_lib = dlopen(g_rpc_lib_path, RTLD_NOW | RTLD_GLOBAL);
        if (g_rpcmem_lib) {
            fprintf(stderr, "[server] rpcmem: loaded from --rpc_lib: %s\n", g_rpc_lib_path);
        } else {
            fprintf(stderr, "[server] rpcmem: dlopen(%s) failed: %s\n", g_rpc_lib_path, dlerror());
        }
    }

    /* 2. Same directory as backend_path (<backend_dir>/libcdsprpc.so) */
    if (!g_rpcmem_lib && backend_path) {
        const char* slash = strrchr(backend_path, '/');
        if (slash) {
            size_t dir_len = (size_t)(slash - backend_path);
            if (dir_len < sizeof(path) - 32) {
                memcpy(path, backend_path, dir_len);
                path[dir_len] = '\0';
                strcat(path, "/libcdsprpc.so");
                g_rpcmem_lib = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
                if (g_rpcmem_lib) {
                    fprintf(stderr, "[server] rpcmem: loaded %s\n", path);
                }
            }
        }
    }

    /* 3. Base directory lib (<base_dir>/lib/libcdsprpc.so) */
    if (!g_rpcmem_lib && base_dir) {
        snprintf(path, sizeof(path), "%s/lib/libcdsprpc.so", base_dir);
        g_rpcmem_lib = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
        if (g_rpcmem_lib) {
            fprintf(stderr, "[server] rpcmem: loaded %s\n", path);
        }
    }

    /* 4. Candidate application internal payload paths */
    if (!g_rpcmem_lib) {
        const char* app_paths[] = {
            "/data/local/tmp/sdxl_app_engine/lib/libcdsprpc.so",
            "/data/user/0/com.sdxlnpu.app/files/termux_bundle/runtime_payload/lib/libcdsprpc.so",
            "/data/data/com.sdxlnpu.app/files/termux_bundle/runtime_payload/lib/libcdsprpc.so",
            "/data/local/tmp/sdxl_test/lib/libcdsprpc.so",
            NULL
        };
        for (int i = 0; !g_rpcmem_lib && app_paths[i]; ++i) {
            if (access(app_paths[i], R_OK) == 0) {
                g_rpcmem_lib = dlopen(app_paths[i], RTLD_NOW | RTLD_GLOBAL);
                if (g_rpcmem_lib) {
                    fprintf(stderr, "[server] rpcmem: loaded %s\n", app_paths[i]);
                }
            }
        }
    }

    /* 5. System dynamic linker resolution */
    if (!g_rpcmem_lib) {
        g_rpcmem_lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
        if (g_rpcmem_lib) {
            fprintf(stderr, "[server] rpcmem: loaded libcdsprpc.so via system linker\n");
        }
    }

    /* 6. Vendor fallbacks */
    if (!g_rpcmem_lib) {
        const char* vendor_paths[] = {
            "/vendor/lib64/libcdsprpc.so",
            "/system/vendor/lib64/libcdsprpc.so",
            "/system/lib64/libcdsprpc.so",
            "/vendor/lib64/librpcmem.so",
            "librpcmem.so",
            NULL
        };
        for (int i = 0; !g_rpcmem_lib && vendor_paths[i]; ++i) {
            g_rpcmem_lib = dlopen(vendor_paths[i], RTLD_NOW | RTLD_GLOBAL);
            if (g_rpcmem_lib) {
                fprintf(stderr, "[server] rpcmem: loaded %s\n", vendor_paths[i]);
            }
        }
    }

    if (!g_rpcmem_lib) {
        fprintf(stderr, "[server] rpcmem: not available (%s), using regular malloc\n", dlerror());
        return;
    }
    g_rpcmem_alloc  = (rpcmem_alloc_fn_t)dlsym(g_rpcmem_lib, "rpcmem_alloc");
    g_rpcmem_free   = (rpcmem_free_fn_t)dlsym(g_rpcmem_lib, "rpcmem_free");
    g_rpcmem_to_fd  = (rpcmem_to_fd_fn_t)dlsym(g_rpcmem_lib, "rpcmem_to_fd");
    g_rpcmem_init   = (rpcmem_init_fn_t)dlsym(g_rpcmem_lib, "rpcmem_init");
    g_rpcmem_deinit = (rpcmem_deinit_fn_t)dlsym(g_rpcmem_lib, "rpcmem_deinit");

    if (g_rpcmem_alloc && g_rpcmem_free) {
        if (g_rpcmem_init) g_rpcmem_init();
        g_rpcmem_available = 1;
        fprintf(stderr, "[server] rpcmem: initialized\n");
    } else {
        fprintf(stderr, "[server] rpcmem: symbols not found\n");
    }
}

static void* shared_alloc(size_t size) {
    if (g_rpcmem_available && g_rpcmem_alloc) {
        void* p = g_rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)size);
        if (p) return p;
        fprintf(stderr, "[server] rpcmem_alloc failed for %zu bytes, fallback to calloc\n", size);
    }
    return calloc(1, size);
}

static void shared_free(void* p) {
    if (g_rpcmem_available && g_rpcmem_free) {
        g_rpcmem_free(p);
    } else {
        free(p);
    }
}

static int shared_to_fd(void* p) {
    if (g_rpcmem_available && g_rpcmem_to_fd) {
        return g_rpcmem_to_fd(p);
    }
    return -1;
}

/* ========================================================================= */
/*  Constants                                                                */
/* ========================================================================= */

#define MAX_CONTEXTS     16
#define MAX_TENSORS      32
#define MAX_ID_LEN       128
#define MAX_PATH_LEN     1024
#define MAX_LINE_LEN     4096
#define MAX_GRAPH_NAME   256

/* ========================================================================= */
/*  Data types                                                               */
/* ========================================================================= */

typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn)(
    const QnnInterface_t*** providerList, uint32_t* numProviders);

typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn)(
    const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

typedef struct {
    char     id[MAX_ID_LEN];
    char     binaryPath[MAX_PATH_LEN];
    int      active;

    /* context binary (mmap'd) */
    void*    binaryData;
    size_t   binarySize;
    int      binaryFd;

    /* QNN handles */
    Qnn_ContextHandle_t contextHandle;
    Qnn_GraphHandle_t   graphHandle;
    char                graphName[MAX_GRAPH_NAME];

    /* input tensors */
    uint32_t     numInputs;
    Qnn_Tensor_t inputs[MAX_TENSORS];
    uint32_t*    inputDims[MAX_TENSORS];   /* owned copies of dimension arrays */
    void*        inputBufs[MAX_TENSORS];   /* pre-allocated data buffers */
    size_t       inputBufSizes[MAX_TENSORS];
    char         inputNames[MAX_TENSORS][MAX_GRAPH_NAME];
    Qnn_MemHandle_t inputMemHandles[MAX_TENSORS];

    /* output tensors */
    uint32_t     numOutputs;
    Qnn_Tensor_t outputs[MAX_TENSORS];
    uint32_t*    outputDims[MAX_TENSORS];
    void*        outputBufs[MAX_TENSORS];
    size_t       outputBufSizes[MAX_TENSORS];
    char         outputNames[MAX_TENSORS][MAX_GRAPH_NAME];
    Qnn_MemHandle_t outputMemHandles[MAX_TENSORS];

    size_t       modelBytes;
} ContextSlot;

/* ========================================================================= */
/*  Globals                                                                  */
/* ========================================================================= */

static void*  g_backendLib   = NULL;
static void*  g_systemLib    = NULL;

/* interface function pointer tables */
static QNN_INTERFACE_VER_TYPE        g_qnn;
static QNN_SYSTEM_INTERFACE_VER_TYPE g_sys;

static Qnn_LogHandle_t     g_logHandle     = NULL;
static Qnn_BackendHandle_t g_backendHandle = NULL;
static Qnn_DeviceHandle_t  g_deviceHandle  = NULL;

static ContextSlot g_slots[MAX_CONTEXTS];
static int         g_numSlots = 0;
static size_t      g_totalLoadedBytes = 0;

/* ========================================================================= */
/*  Logging                                                                  */
/* ========================================================================= */

static void qnn_log_callback(const char* fmt,
                              QnnLog_Level_t level,
                              uint64_t timestamp,
                              va_list args) {
    (void)timestamp;
    if (level > QNN_LOG_LEVEL_WARN) return;
    fprintf(stderr, "[QNN] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
}

/* ========================================================================= */
/*  Helpers                                                                  */
/* ========================================================================= */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static size_t datatype_size(Qnn_DataType_t dt) {
    switch (dt) {
        case QNN_DATATYPE_FLOAT_64:
        case QNN_DATATYPE_INT_64:
        case QNN_DATATYPE_UINT_64:
            return 8;
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:
        case QNN_DATATYPE_SFIXED_POINT_32:
        case QNN_DATATYPE_UFIXED_POINT_32:
            return 4;
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_BFLOAT_16:
        case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_SFIXED_POINT_16:
        case QNN_DATATYPE_UFIXED_POINT_16:
            return 2;
        case QNN_DATATYPE_FLOAT_8:
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8:
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_BOOL_8:
            return 1;
        default:
            return 1;
    }
}

static size_t calc_tensor_bytes(uint32_t rank, const uint32_t* dims, Qnn_DataType_t dt) {
    if (rank == 0 || dims == NULL) return 0;
    size_t elems = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        elems *= dims[i];
    }
    return elems * datatype_size(dt);
}

/* read entire file into malloc'd buffer (QNN may need writable memory for relocations) */
static int load_file_malloc(const char* path, void** out_data, size_t* out_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    size_t sz = (size_t)st.st_size;
    void* p = malloc(sz);
    if (!p) { close(fd); return -1; }
    size_t total = 0;
    while (total < sz) {
        ssize_t r = read(fd, (uint8_t*)p + total, sz - total);
        if (r <= 0) { free(p); close(fd); return -1; }
        total += (size_t)r;
    }
    close(fd);
    *out_data = p;
    *out_size = sz;
    return 0;
}

static int __attribute__((unused)) read_file_to_buf(const char* path, void* buf, size_t expected_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    size_t rd = fread(buf, 1, expected_size, f);
    fclose(f);
    return (rd == expected_size) ? 0 : -1;
}

static int write_raw_file(const char* path, const void* data, size_t size) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    size_t wr = fwrite(data, 1, size, f);
    fclose(f);
    return (wr == size) ? 0 : -1;
}

static int mkdirs(const char* path) {
    char tmp[MAX_PATH_LEN];
    size_t len = strlen(path);
    if (len >= MAX_PATH_LEN) return -1;
    memcpy(tmp, path, len + 1);
    for (size_t i = 1; i < len; ++i) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            mkdir(tmp, 0755);
            tmp[i] = '/';
        }
    }
    return mkdir(tmp, 0755);
}

/* find a slot by id, return index or -1 */
static int find_slot(const char* id) {
    for (int i = 0; i < g_numSlots; ++i) {
        if (g_slots[i].active && strcmp(g_slots[i].id, id) == 0) return i;
    }
    return -1;
}

/* ========================================================================= */
/*  QNN initialization                                                       */
/* ========================================================================= */

static int init_qnn(const char* backend_path, const char* system_path, const char* base_dir) {
    setup_dsp_environment(backend_path, base_dir);
    init_rpcmem(backend_path, base_dir);
    /* Load backend library */
    g_backendLib = dlopen(backend_path, RTLD_NOW | RTLD_LOCAL);
    if (!g_backendLib) {
        fprintf(stderr, "ERR: dlopen backend: %s\n", dlerror());
        return -1;
    }

    QnnInterfaceGetProvidersFn getProviders =
        (QnnInterfaceGetProvidersFn)dlsym(g_backendLib, "QnnInterface_getProviders");
    if (!getProviders) {
        fprintf(stderr, "ERR: dlsym QnnInterface_getProviders: %s\n", dlerror());
        return -1;
    }

    const QnnInterface_t** providers = NULL;
    uint32_t numProviders = 0;
    if (QNN_SUCCESS != getProviders(&providers, &numProviders) || numProviders == 0) {
        fprintf(stderr, "ERR: QnnInterface_getProviders failed\n");
        return -1;
    }
    g_qnn = providers[0]->QNN_INTERFACE_VER_NAME;

    /* Load system library */
    g_systemLib = dlopen(system_path, RTLD_NOW | RTLD_LOCAL);
    if (!g_systemLib) {
        fprintf(stderr, "ERR: dlopen system: %s\n", dlerror());
        return -1;
    }

    QnnSystemInterfaceGetProvidersFn getSysProviders =
        (QnnSystemInterfaceGetProvidersFn)dlsym(g_systemLib, "QnnSystemInterface_getProviders");
    if (!getSysProviders) {
        fprintf(stderr, "ERR: dlsym QnnSystemInterface_getProviders: %s\n", dlerror());
        return -1;
    }

    const QnnSystemInterface_t** sysProviders = NULL;
    uint32_t numSysProviders = 0;
    if (QNN_SUCCESS != getSysProviders(&sysProviders, &numSysProviders) || numSysProviders == 0) {
        fprintf(stderr, "ERR: QnnSystemInterface_getProviders failed\n");
        return -1;
    }
    g_sys = sysProviders[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    /* Create log handle */
    if (g_qnn.logCreate) {
        g_qnn.logCreate(qnn_log_callback, QNN_LOG_LEVEL_ERROR, &g_logHandle);
    }

    /* Create backend */
    if (!g_qnn.backendCreate) {
        fprintf(stderr, "ERR: backendCreate is NULL\n");
        return -1;
    }
    Qnn_ErrorHandle_t err = g_qnn.backendCreate(g_logHandle, NULL, &g_backendHandle);
    if (QNN_SUCCESS != err) {
        fprintf(stderr, "ERR: backendCreate failed: %d\n", (int)err);
        return -1;
    }

    /* Create device */
    if (g_qnn.propertyHasCapability && g_qnn.deviceCreate) {
        err = g_qnn.deviceCreate(g_logHandle, NULL, &g_deviceHandle);
        if (QNN_SUCCESS != err) {
            fprintf(stderr, "WARN: deviceCreate failed: %d (continuing without device)\n", (int)err);
            g_deviceHandle = NULL;
        }
    }

    return 0;
}

static uint32_t g_powerConfigId = 0;

static void set_perf_mode(void) {
    if (!g_qnn.propertyHasCapability || !g_qnn.deviceGetInfrastructure) return;

    Qnn_ErrorHandle_t propErr = g_qnn.propertyHasCapability(QNN_PROPERTY_DEVICE_SUPPORT_INFRASTRUCTURE);
    if (QNN_PROPERTY_SUPPORTED != propErr && QNN_SUCCESS != propErr) return;

    QnnDevice_Infrastructure_t infraOpaque = NULL;
    if (QNN_SUCCESS != g_qnn.deviceGetInfrastructure(&infraOpaque) || !infraOpaque) return;

    const QnnHtpDevice_Infrastructure_t* htpInfra =
        (const QnnHtpDevice_Infrastructure_t*)infraOpaque;
    if (htpInfra->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
    if (!htpInfra->perfInfra.createPowerConfigId || !htpInfra->perfInfra.setPowerConfig) return;

    /* Create a power config ID */
    Qnn_ErrorHandle_t err = htpInfra->perfInfra.createPowerConfigId(0, 0, &g_powerConfigId);
    if (QNN_SUCCESS != err) {
        fprintf(stderr, "[server] WARN: createPowerConfigId failed: %d\n", (int)err);
        return;
    }

    QnnHtpPerfInfrastructure_PowerConfig_t dcvs;
    memset(&dcvs, 0, sizeof(dcvs));
    dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    dcvs.dcvsV3Config.contextId = g_powerConfigId;
    dcvs.dcvsV3Config.setDcvsEnable = 1;
    dcvs.dcvsV3Config.dcvsEnable = 0;  /* burst: disable DCVS, pin to max freq */
    dcvs.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    dcvs.dcvsV3Config.setSleepDisable = 1;
    dcvs.dcvsV3Config.sleepDisable = 1;
    dcvs.dcvsV3Config.setBusParams = 1;
    dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.dcvsV3Config.setCoreParams = 1;
    dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    /* Add RPC control latency (0us) and polling time (9999us) for true burst */
    QnnHtpPerfInfrastructure_PowerConfig_t rpc_lat;
    memset(&rpc_lat, 0, sizeof(rpc_lat));
    rpc_lat.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpc_lat.rpcControlLatencyConfig = 0;

    QnnHtpPerfInfrastructure_PowerConfig_t rpc_poll;
    memset(&rpc_poll, 0, sizeof(rpc_poll));
    rpc_poll.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpc_poll.rpcPollingTimeConfig = 9999;

    /* Hexagon V75/V79: Dedicated HMX_V2 clock/voltage voting (force CLK_PERF_HIGH for matrix engine) */
    QnnHtpPerfInfrastructure_PowerConfig_t hmx_cfg;
    memset(&hmx_cfg, 0, sizeof(hmx_cfg));
    hmx_cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2;
    hmx_cfg.hmxV2Config.hmxPickDefault = 0; /* Manual clock vote */
    hmx_cfg.hmxV2Config.hmxVoltageCornerMin = DCVS_EXP_VCORNER_TUR;
    hmx_cfg.hmxV2Config.hmxVoltageCornerTarget = DCVS_EXP_VCORNER_MAX;
    hmx_cfg.hmxV2Config.hmxVoltageCornerMax = DCVS_EXP_VCORNER_MAX;
    hmx_cfg.hmxV2Config.hmxPerfMode = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;

    /* Try with HMX_V2 turbo pin first */
    const QnnHtpPerfInfrastructure_PowerConfig_t* configs_hmx[] = { &dcvs, &rpc_lat, &rpc_poll, &hmx_cfg, NULL };
    err = htpInfra->perfInfra.setPowerConfig(g_powerConfigId, configs_hmx);
    if (QNN_SUCCESS == err) {
        fprintf(stderr, "[server] HTP performance mode set with HMX_V2 turbo (powerConfigId=%u)\n", g_powerConfigId);
        return;
    }

    /* Fallback without HMX_V2 if device/firmware rejects option 5 */
    const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = { &dcvs, &rpc_lat, &rpc_poll, NULL };
    err = htpInfra->perfInfra.setPowerConfig(g_powerConfigId, configs);
    if (QNN_SUCCESS == err) {
        fprintf(stderr, "[server] HTP performance mode set (standard fallback, powerConfigId=%u)\n", g_powerConfigId);
    } else {
        fprintf(stderr, "[server] WARN: setPowerConfig failed: %d\n", (int)err);
    }
}

/* ========================================================================= */
/*  Context loading                                                          */
/* ========================================================================= */

/* Extract graph info from context binary via system API */
static int __attribute__((unused)) get_graph_info_from_binary(const void* data, size_t size,
                                       const char** out_graph_name,
                                       const QnnSystemContext_GraphInfo_t** out_graphs,
                                       uint32_t* out_num_graphs) {
    QnnSystemContext_Handle_t sysCtx = NULL;
    if (QNN_SUCCESS != g_sys.systemContextCreate(&sysCtx)) {
        fprintf(stderr, "ERR: systemContextCreate failed\n");
        return -1;
    }

    const QnnSystemContext_BinaryInfo_t* binInfo = NULL;
    Qnn_ContextBinarySize_t binInfoSize = 0;
    Qnn_ErrorHandle_t err = g_sys.systemContextGetBinaryInfo(
        sysCtx, (void*)data, (uint64_t)size, &binInfo, &binInfoSize);
    if (QNN_SUCCESS != err) {
        fprintf(stderr, "ERR: systemContextGetBinaryInfo failed: %d\n", (int)err);
        g_sys.systemContextFree(sysCtx);
        return -1;
    }

    /* Extract graphs from versioned binary info */
    uint32_t numGraphs = 0;
    const QnnSystemContext_GraphInfo_t* graphs = NULL;

    if (binInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        numGraphs = binInfo->contextBinaryInfoV1.numGraphs;
        graphs = binInfo->contextBinaryInfoV1.graphs;
    } else if (binInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        numGraphs = binInfo->contextBinaryInfoV2.numGraphs;
        graphs = binInfo->contextBinaryInfoV2.graphs;
    } else {
        /* try V3 or newer - has same layout for our fields */
        numGraphs = binInfo->contextBinaryInfoV2.numGraphs;
        graphs = binInfo->contextBinaryInfoV2.graphs;
    }

    *out_graphs = graphs;
    *out_num_graphs = numGraphs;

    if (numGraphs > 0) {
        const QnnSystemContext_GraphInfo_t* g0 = &graphs[0];
        if (g0->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
            *out_graph_name = g0->graphInfoV1.graphName;
        } else if (g0->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
            *out_graph_name = g0->graphInfoV2.graphName;
        } else {
            *out_graph_name = g0->graphInfoV3.graphName;
        }
    }

    /* NOTE: sysCtx owns the binInfo data - we must extract what we need before freeing.
       But we need graphName and tensor info alive during setup. We'll copy what we need
       and free sysCtx after the copy. So we do NOT free here - caller handles it via
       return of sysCtx. Actually, let's just copy everything we need. */

    /* We'll return and let caller use the data, then free sysCtx after */
    /* Actually, to keep it simple: just keep sysCtx alive until we're done setting up.
       The caller will free it. For simplicity, embed the copy logic here. */

    /* Free will happen in the load_context function after we've copied everything */
    /* Store sysCtx handle temporarily - but we can't return it from here easily.
       Let's change approach: extract and copy everything here, then free. */

    /* OK, let me NOT free sysCtx here - I'll return it via an output param */
    /* Actually, re-reading the code, I realize I should just extract everything now */
    return 0; /* caller must free sysCtx */
}

static void setup_tensor_from_info(Qnn_Tensor_t* dst, const Qnn_Tensor_t* src,
                                    uint32_t** out_dims, void** out_buf,
                                    size_t* out_buf_size, char* out_name) {
    /* Initialize to clean state first (like QNN_TENSOR_INIT) */
    memset(dst, 0, sizeof(Qnn_Tensor_t));
    dst->version = src->version;

    /* Extract source fields via version-aware access */
    const char* srcName;
    uint32_t srcId, srcRank;
    Qnn_TensorType_t srcType;
    Qnn_TensorDataFormat_t srcDataFormat;
    Qnn_DataType_t srcDataType;
    Qnn_QuantizeParams_t srcQParams;
    const uint32_t* srcDims;

    if (src->version == QNN_TENSOR_VERSION_2) {
        srcName = src->v2.name;
        srcId = src->v2.id;
        srcType = src->v2.type;
        srcDataFormat = src->v2.dataFormat;
        srcDataType = src->v2.dataType;
        srcQParams = src->v2.quantizeParams;
        srcRank = src->v2.rank;
        srcDims = src->v2.dimensions;
    } else {
        srcName = src->v1.name;
        srcId = src->v1.id;
        srcType = src->v1.type;
        srcDataFormat = src->v1.dataFormat;
        srcDataType = src->v1.dataType;
        srcQParams = src->v1.quantizeParams;
        srcRank = src->v1.rank;
        srcDims = src->v1.dimensions;
    }

    /* Copy name */
    if (srcName) {
        strncpy(out_name, srcName, MAX_GRAPH_NAME - 1);
        out_name[MAX_GRAPH_NAME - 1] = '\0';
    } else {
        snprintf(out_name, MAX_GRAPH_NAME, "tensor_%u", srcId);
    }

    /* Copy dimensions */
    uint32_t* dims = (uint32_t*)malloc(srcRank * sizeof(uint32_t));
    if (dims && srcRank > 0 && srcDims) {
        memcpy(dims, srcDims, srcRank * sizeof(uint32_t));
    }
    *out_dims = dims;

    /* Calculate buffer size and allocate via rpcmem for DMA */
    size_t buf_size = calc_tensor_bytes(srcRank, dims, srcDataType);
    void* buf = shared_alloc(buf_size > 0 ? buf_size : 1);
    if (buf && buf_size > 0) memset(buf, 0, buf_size);
    *out_buf = buf;
    *out_buf_size = buf_size;

    /* Set fields on destination tensor (selective copy, not memcpy) */
    if (dst->version == QNN_TENSOR_VERSION_2) {
        dst->v2.name = out_name;
        dst->v2.id = srcId;
        dst->v2.type = srcType;
        dst->v2.dataFormat = srcDataFormat;
        dst->v2.dataType = srcDataType;
        dst->v2.quantizeParams = srcQParams;
        dst->v2.rank = srcRank;
        dst->v2.dimensions = dims;
        dst->v2.memType = QNN_TENSORMEMTYPE_RAW;
        dst->v2.clientBuf.data = buf;
        dst->v2.clientBuf.dataSize = (uint32_t)buf_size;
    } else {
        dst->v1.name = out_name;
        dst->v1.id = srcId;
        dst->v1.type = srcType;
        dst->v1.dataFormat = srcDataFormat;
        dst->v1.dataType = srcDataType;
        dst->v1.quantizeParams = srcQParams;
        dst->v1.rank = srcRank;
        dst->v1.dimensions = dims;
        dst->v1.memType = QNN_TENSORMEMTYPE_RAW;
        dst->v1.clientBuf.data = buf;
        dst->v1.clientBuf.dataSize = (uint32_t)buf_size;
    }
}

/* Register a tensor's rpcmem buffer with QNN for DMA access */
static int register_tensor_mem(Qnn_ContextHandle_t ctx, Qnn_Tensor_t* tensor,
                                void* buf, size_t buf_size, Qnn_MemHandle_t* out_handle) {
    *out_handle = NULL;
    if (!g_rpcmem_available || !buf || buf_size == 0) return 0; /* skip if no rpcmem */

    int fd = shared_to_fd(buf);
    if (fd < 0) {
        fprintf(stderr, "[server] rpcmem_to_fd failed, skipping mem registration\n");
        return 0; /* non-fatal, will use clientBuf fallback */
    }
    fprintf(stderr, "[server] register_tensor_mem: fd=%d buf=%p size=%zu\n", fd, buf, buf_size);

    Qnn_MemDescriptor_t memDesc;
    memset(&memDesc, 0, sizeof(memDesc));
    memDesc.memShape.numDim = 1;
    uint32_t flatDim = (uint32_t)buf_size;
    memDesc.memShape.dimSize = &flatDim;
    memDesc.dataType = QNN_DATATYPE_UINT_8;
    /* Try ION registration */
    memDesc.memType = QNN_MEM_TYPE_ION;
    memDesc.ionInfo.fd = fd;

    Qnn_MemHandle_t handle = NULL;
    Qnn_ErrorHandle_t err = g_qnn.memRegister(ctx, &memDesc, 1, &handle);
    if (QNN_SUCCESS != err) {
        fprintf(stderr, "[server] ION memRegister failed: %d (fd=%d, size=%zu)\n",
                (int)err, fd, buf_size);
        return 0; /* non-fatal */
    }

    *out_handle = handle;

    /* Switch tensor from clientBuf (RAW) to memHandle */
    if (tensor->version == QNN_TENSOR_VERSION_2) {
        tensor->v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tensor->v2.memHandle = handle;
    } else {
        tensor->v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        tensor->v1.memHandle = handle;
    }
    return 1;
}

static int cmd_load(const char* id, const char* context_path) {
    if (find_slot(id) >= 0) {
        printf("ERR already_loaded %s\n", id);
        fflush(stdout);
        return -1;
    }
    if (g_numSlots >= MAX_CONTEXTS) {
        printf("ERR max_contexts_reached\n");
        fflush(stdout);
        return -1;
    }

    ContextSlot* slot = &g_slots[g_numSlots];
    memset(slot, 0, sizeof(ContextSlot));
    strncpy(slot->id, id, MAX_ID_LEN - 1);
    strncpy(slot->binaryPath, context_path, MAX_PATH_LEN - 1);

    /* read the context binary into malloc'd buffer */
    if (load_file_malloc(context_path, &slot->binaryData, &slot->binarySize) != 0) {
        printf("ERR cannot_open %s: %s\n", context_path, strerror(errno));
        fflush(stdout);
        return -1;
    }
    slot->modelBytes = slot->binarySize;

    if (g_totalLoadedBytes + slot->modelBytes > 3758096384ULL) {
        fprintf(stderr, "[server] WARN: loading %s (%.1f MB) pushes total DSP context size (%.1f MB) past 3.5 GB FastRPC budget!\n",
                id, (double)slot->modelBytes / (1024.0 * 1024.0),
                (double)(g_totalLoadedBytes + slot->modelBytes) / (1024.0 * 1024.0));
    }

    fprintf(stderr, "[server] Loading context %s: %s (%.1f MB)\n",
            id, context_path, (double)slot->binarySize / (1024.0 * 1024.0));

    /* Get graph metadata via system API */
    QnnSystemContext_Handle_t sysCtx = NULL;
    if (QNN_SUCCESS != g_sys.systemContextCreate(&sysCtx)) {
        printf("ERR systemContextCreate_failed\n");
        fflush(stdout);
        free(slot->binaryData);
        return -1;
    }

    const QnnSystemContext_BinaryInfo_t* binInfo = NULL;
    Qnn_ContextBinarySize_t binInfoSize = 0;
    Qnn_ErrorHandle_t err = g_sys.systemContextGetBinaryInfo(
        sysCtx, slot->binaryData, (uint64_t)slot->binarySize, &binInfo, &binInfoSize);
    if (QNN_SUCCESS != err) {
        printf("ERR getBinaryInfo_failed %d\n", (int)err);
        fflush(stdout);
        g_sys.systemContextFree(sysCtx);
        free(slot->binaryData);
        return -1;
    }

    fprintf(stderr, "[server] BinaryInfo version=%u\n", binInfo->version);

    /* Extract graph info from versioned binary info */
    uint32_t numGraphs = 0;
    const QnnSystemContext_GraphInfo_t* graphs = NULL;

    if (binInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        numGraphs = binInfo->contextBinaryInfoV1.numGraphs;
        graphs = binInfo->contextBinaryInfoV1.graphs;
    } else if (binInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        numGraphs = binInfo->contextBinaryInfoV2.numGraphs;
        graphs = binInfo->contextBinaryInfoV2.graphs;
    } else {
        numGraphs = binInfo->contextBinaryInfoV3.numGraphs;
        graphs = binInfo->contextBinaryInfoV3.graphs;
    }

    fprintf(stderr, "[server] numGraphs=%u graphs=%p\n", numGraphs, (void*)graphs);

    if (numGraphs == 0) {
        printf("ERR no_graphs_in_context\n");
        fflush(stdout);
        g_sys.systemContextFree(sysCtx);
        free(slot->binaryData);
        return -1;
    }

    /* Use first graph */
    const QnnSystemContext_GraphInfo_t* gi = &graphs[0];
    fprintf(stderr, "[server] GraphInfo version=%u numGraphs=%u\n", gi->version, numGraphs);

    /* Debug: print all graph names */
    for (uint32_t dbg = 0; dbg < numGraphs; ++dbg) {
        const QnnSystemContext_GraphInfo_t* gdbg = &graphs[dbg];
        const char* gn = NULL;
        if (gdbg->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
            gn = gdbg->graphInfoV1.graphName;
        } else if (gdbg->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
            gn = gdbg->graphInfoV2.graphName;
        } else {
            gn = gdbg->graphInfoV3.graphName;
        }
        fprintf(stderr, "[server]   graph[%u]: name='%s' version=%u\n",
                dbg, gn ? gn : "(null)", gdbg->version);
    }
    const char* graphName = NULL;
    uint32_t numInputs = 0, numOutputs = 0;
    const Qnn_Tensor_t* graphInputs = NULL;
    const Qnn_Tensor_t* graphOutputs = NULL;

    if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        graphName = gi->graphInfoV1.graphName;
        numInputs = gi->graphInfoV1.numGraphInputs;
        numOutputs = gi->graphInfoV1.numGraphOutputs;
        graphInputs = gi->graphInfoV1.graphInputs;
        graphOutputs = gi->graphInfoV1.graphOutputs;
    } else if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        graphName = gi->graphInfoV2.graphName;
        numInputs = gi->graphInfoV2.numGraphInputs;
        numOutputs = gi->graphInfoV2.numGraphOutputs;
        graphInputs = gi->graphInfoV2.graphInputs;
        graphOutputs = gi->graphInfoV2.graphOutputs;
    } else {
        graphName = gi->graphInfoV3.graphName;
        numInputs = gi->graphInfoV3.numGraphInputs;
        numOutputs = gi->graphInfoV3.numGraphOutputs;
        graphInputs = gi->graphInfoV3.graphInputs;
        graphOutputs = gi->graphInfoV3.graphOutputs;
    }

    if (graphName) {
        strncpy(slot->graphName, graphName, MAX_GRAPH_NAME - 1);
    }

    if (numInputs > MAX_TENSORS) numInputs = MAX_TENSORS;
    if (numOutputs > MAX_TENSORS) numOutputs = MAX_TENSORS;
    slot->numInputs = numInputs;
    slot->numOutputs = numOutputs;

    /* Copy tensor metadata and allocate buffers */
    for (uint32_t i = 0; i < numInputs; ++i) {
        setup_tensor_from_info(&slot->inputs[i], &graphInputs[i],
                               &slot->inputDims[i], &slot->inputBufs[i],
                               &slot->inputBufSizes[i], slot->inputNames[i]);
        /* Mark as app-writable for graphExecute */
        if (slot->inputs[i].version == QNN_TENSOR_VERSION_2) {
            slot->inputs[i].v2.type = QNN_TENSOR_TYPE_APP_WRITE;
        } else {
            slot->inputs[i].v1.type = QNN_TENSOR_TYPE_APP_WRITE;
        }
        fprintf(stderr, "[server] input[%u] name='%s' dtype=0x%04x size=%zu\n",
                i, slot->inputNames[i],
                (unsigned)(slot->inputs[i].version == QNN_TENSOR_VERSION_2
                    ? slot->inputs[i].v2.dataType : slot->inputs[i].v1.dataType),
                slot->inputBufSizes[i]);
    }
    for (uint32_t i = 0; i < numOutputs; ++i) {
        setup_tensor_from_info(&slot->outputs[i], &graphOutputs[i],
                               &slot->outputDims[i], &slot->outputBufs[i],
                               &slot->outputBufSizes[i], slot->outputNames[i]);
        /* Mark as app-readable for graphExecute */
        if (slot->outputs[i].version == QNN_TENSOR_VERSION_2) {
            slot->outputs[i].v2.type = QNN_TENSOR_TYPE_APP_READ;
        } else {
            slot->outputs[i].v1.type = QNN_TENSOR_TYPE_APP_READ;
        }
        fprintf(stderr, "[server] output[%u] name='%s' dtype=0x%04x size=%zu\n",
                i, slot->outputNames[i],
                (unsigned)(slot->outputs[i].version == QNN_TENSOR_VERSION_2
                    ? slot->outputs[i].v2.dataType : slot->outputs[i].v1.dataType),
                slot->outputBufSizes[i]);
    }

    /* Free system context - we've copied everything we need */
    g_sys.systemContextFree(sysCtx);

    /* Create QNN context from binary */
    double t0 = now_ms();
    err = g_qnn.contextCreateFromBinary(
        g_backendHandle, g_deviceHandle, NULL,
        slot->binaryData, (Qnn_ContextBinarySize_t)slot->binarySize,
        &slot->contextHandle, NULL);
    if (QNN_SUCCESS != err) {
        printf("ERR contextCreateFromBinary_failed %d\n", (int)err);
        fflush(stdout);
        free(slot->binaryData);
        return -1;
    }
    double t1 = now_ms();

    /* Retrieve graph handle */
    err = g_qnn.graphRetrieve(slot->contextHandle, slot->graphName, &slot->graphHandle);
    if (QNN_SUCCESS != err) {
        printf("ERR graphRetrieve_failed %d graph=%s\n", (int)err, slot->graphName);
        fflush(stdout);
        g_qnn.contextFree(slot->contextHandle, NULL);
        free(slot->binaryData);
        return -1;
    }

    /* Unlock maximum HVX hardware threads on Hexagon V79 (8 or 6 instead of default 4) prior to first execution */
    if (g_qnn.graphSetConfig) {
        const uint64_t try_threads[2] = {8, 6};
        for (int ti = 0; ti < 2; ++ti) {
            QnnHtpGraph_CustomConfig_t htpCfg;
            memset(&htpCfg, 0, sizeof(htpCfg));
            htpCfg.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
            htpCfg.numHvxThreads = try_threads[ti];

            QnnGraph_Config_t gCfg;
            memset(&gCfg, 0, sizeof(gCfg));
            gCfg.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            gCfg.customConfig = &htpCfg;
            const QnnGraph_Config_t* cfgList[2] = {&gCfg, NULL};

            if (QNN_SUCCESS == g_qnn.graphSetConfig(slot->graphHandle, cfgList)) {
                fprintf(stderr, "[server] Configured graph '%s' to use %llu HVX threads\n",
                        slot->graphName, (unsigned long long)try_threads[ti]);
                break;
            }
        }
    }

    /* Finalize deserialized graph if supported by backend */
    if (g_qnn.graphFinalize && g_qnn.propertyHasCapability) {
        Qnn_ErrorHandle_t propErr = g_qnn.propertyHasCapability(
            QNN_PROPERTY_GRAPH_SUPPORT_FINALIZE_DESERIALIZED_GRAPH);
        fprintf(stderr, "[server] finalize_deserialized property: %d (SUPPORTED=%d)\n",
                (int)propErr, (int)QNN_PROPERTY_SUPPORTED);
        if (QNN_PROPERTY_SUPPORTED == propErr) {
            err = g_qnn.graphFinalize(slot->graphHandle, NULL, NULL);
            if (QNN_SUCCESS != err) {
                fprintf(stderr, "[server] WARN: graphFinalize failed: %d (continuing)\n", (int)err);
            } else {
                fprintf(stderr, "[server] graphFinalize OK for %s\n", slot->graphName);
            }
        }
    }

    double t2 = now_ms();

    /* Free the binary data — no longer needed after contextCreate + graphRetrieve */
    free(slot->binaryData);
    slot->binaryData = NULL;
    slot->binarySize = 0;

    /* Register rpcmem buffers with QNN for DMA */
    if (g_rpcmem_available) {
        int reg_count = 0;
        for (uint32_t i = 0; i < numInputs; ++i) {
            int r = register_tensor_mem(slot->contextHandle, &slot->inputs[i],
                                              slot->inputBufs[i], slot->inputBufSizes[i],
                                              &slot->inputMemHandles[i]);
            fprintf(stderr, "[server] reg input[%u] -> %d\n", i, r);
            reg_count += r;
        }
        for (uint32_t i = 0; i < numOutputs; ++i) {
            int r = register_tensor_mem(slot->contextHandle, &slot->outputs[i],
                                              slot->outputBufs[i], slot->outputBufSizes[i],
                                              &slot->outputMemHandles[i]);
            fprintf(stderr, "[server] reg output[%u] -> %d\n", i, r);
            reg_count += r;
        }
        fprintf(stderr, "[server] Registered %d/%u tensor buffers with QNN\n",
                reg_count, numInputs + numOutputs);
    }

    g_totalLoadedBytes += slot->modelBytes;
    slot->active = 1;
    g_numSlots++;

    fprintf(stderr, "[server] Context %s loaded: graph=%s inputs=%u outputs=%u "
                    "ctx=%.0fms retrieve=%.0fms\n",
            id, slot->graphName, numInputs, numOutputs, t1 - t0, t2 - t1);

    printf("OK %s %u %u\n", slot->graphName, numInputs, numOutputs);
    fflush(stdout);
    return 0;
}

/* ========================================================================= */
/*  Running inference                                                        */
/* ========================================================================= */

/*
 * Parse a single input-list line into slot's input buffers.
 * Format: "file1.raw file2.raw ..." or "name:=file1.raw name2:=file2.raw"
 * Input .raw files contain float32 data (from numpy). Converted to native dtype.
 */
static int parse_input_line(char* line, ContextSlot* slot) {
    char* saveptr = NULL;
    char* tok = strtok_r(line, " \t", &saveptr);
    uint32_t idx = 0;

    while (tok && idx < slot->numInputs) {
        char* eq = strstr(tok, ":=");
        const char* filepath;
        int target_idx = -1;

        if (eq) {
            *eq = '\0';
            const char* name = tok;
            filepath = eq + 2;
            for (uint32_t j = 0; j < slot->numInputs; ++j) {
                if (strcmp(slot->inputNames[j], name) == 0) {
                    target_idx = (int)j;
                    break;
                }
            }
            if (target_idx < 0) {
                fprintf(stderr, "WARN: input tensor name not found: %s\n", name);
                tok = strtok_r(NULL, " \t", &saveptr);
                continue;
            }
        } else {
            filepath = tok;
            target_idx = (int)idx;
        }

        if (target_idx >= 0 && (uint32_t)target_idx < slot->numInputs) {
            Qnn_Tensor_t* t = &slot->inputs[target_idx];
            Qnn_DataType_t dt = (t->version == QNN_TENSOR_VERSION_2)
                ? t->v2.dataType : t->v1.dataType;
            uint32_t rank = (t->version == QNN_TENSOR_VERSION_2)
                ? t->v2.rank : t->v1.rank;
            const uint32_t* dims = (t->version == QNN_TENSOR_VERSION_2)
                ? t->v2.dimensions : t->v1.dimensions;

            size_t numElems = 1;
            for (uint32_t d = 0; d < rank; d++) numElems *= dims[d];

            FILE* fin = fopen(filepath, "rb");
            if (!fin) {
                fprintf(stderr, "ERR: cannot open input %s\n", filepath);
                return -1;
            }

            void* dst = slot->inputBufs[target_idx];
            size_t rd;

            /* Fast path: FLOAT_32 — read directly into input buffer (no temp alloc) */
            if (dt == QNN_DATATYPE_FLOAT_32) {          /* 0x0232 */
                rd = fread(dst, sizeof(float), numElems, fin);
                fclose(fin);
            } else {
                float* floatBuf = (float*)malloc(numElems * sizeof(float));
                rd = fread(floatBuf, sizeof(float), numElems, fin);
                fclose(fin);

                if (dt == QNN_DATATYPE_FLOAT_16) {   /* 0x0216 */
                /* float32 -> IEEE half */
                uint16_t* p = (uint16_t*)dst;
                for (size_t e = 0; e < rd; e++) {
                    uint32_t bits;
                    memcpy(&bits, &floatBuf[e], 4);
                    uint32_t s = (bits >> 16) & 0x8000;
                    int32_t ex = ((bits >> 23) & 0xFF) - 127 + 15;
                    uint32_t m = bits & 0x007FFFFF;
                    if (ex <= 0) p[e] = (uint16_t)s;
                    else if (ex >= 31) p[e] = (uint16_t)(s | 0x7C00);
                    else p[e] = (uint16_t)(s | (ex << 10) | (m >> 13));
                }
            } else if (dt == QNN_DATATYPE_INT_32) {     /* 0x0032 */
                int32_t* p = (int32_t*)dst;
                for (size_t e = 0; e < rd; e++) p[e] = (int32_t)floatBuf[e];
            } else if (dt == QNN_DATATYPE_INT_16) {     /* 0x0016 */
                int16_t* p = (int16_t*)dst;
                for (size_t e = 0; e < rd; e++) p[e] = (int16_t)floatBuf[e];
            } else if (dt == QNN_DATATYPE_INT_8) {      /* 0x0008 */
                int8_t* p = (int8_t*)dst;
                for (size_t e = 0; e < rd; e++) p[e] = (int8_t)floatBuf[e];
            } else if (dt == QNN_DATATYPE_UINT_32) {    /* 0x0132 */
                uint32_t* p = (uint32_t*)dst;
                for (size_t e = 0; e < rd; e++) p[e] = (uint32_t)floatBuf[e];
            } else if (dt == QNN_DATATYPE_UINT_16) {    /* 0x0116 */
                uint16_t* p = (uint16_t*)dst;
                for (size_t e = 0; e < rd; e++) p[e] = (uint16_t)floatBuf[e];
            } else if (dt == QNN_DATATYPE_UINT_8) {     /* 0x0108 */
                uint8_t* p = (uint8_t*)dst;
                for (size_t e = 0; e < rd; e++) p[e] = (uint8_t)floatBuf[e];
            } else if (dt == QNN_DATATYPE_UFIXED_POINT_16) { /* 0x0416 */
                Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
                    ? t->v2.quantizeParams : t->v1.quantizeParams;
                uint16_t* p = (uint16_t*)dst;
                if (qp.encodingDefinition == QNN_DEFINITION_DEFINED &&
                    qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET &&
                    qp.scaleOffsetEncoding.scale != 0.0f) {
                    float inv_scale = 1.0f / qp.scaleOffsetEncoding.scale;
                    int32_t offset = qp.scaleOffsetEncoding.offset;
                    for (size_t e = 0; e < rd; e++) {
                        int32_t q = (int32_t)lrintf(floatBuf[e] * inv_scale) - offset;
                        if (q < 0) q = 0;
                        else if (q > 65535) q = 65535;
                        p[e] = (uint16_t)q;
                    }
                } else {
                    for (size_t e = 0; e < rd; e++) p[e] = (uint16_t)floatBuf[e];
                }
            } else if (dt == QNN_DATATYPE_UFIXED_POINT_8) {  /* 0x0408 */
                Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
                    ? t->v2.quantizeParams : t->v1.quantizeParams;
                uint8_t* p = (uint8_t*)dst;
                if (qp.encodingDefinition == QNN_DEFINITION_DEFINED &&
                    qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET &&
                    qp.scaleOffsetEncoding.scale != 0.0f) {
                    float inv_scale = 1.0f / qp.scaleOffsetEncoding.scale;
                    int32_t offset = qp.scaleOffsetEncoding.offset;
                    for (size_t e = 0; e < rd; e++) {
                        int32_t q = (int32_t)lrintf(floatBuf[e] * inv_scale) - offset;
                        if (q < 0) q = 0;
                        else if (q > 255) q = 255;
                        p[e] = (uint8_t)q;
                    }
                } else {
                    for (size_t e = 0; e < rd; e++) p[e] = (uint8_t)floatBuf[e];
                }
            } else if (dt == QNN_DATATYPE_SFIXED_POINT_16) { /* 0x0316 */
                Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
                    ? t->v2.quantizeParams : t->v1.quantizeParams;
                int16_t* p = (int16_t*)dst;
                if (qp.encodingDefinition == QNN_DEFINITION_DEFINED &&
                    qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET &&
                    qp.scaleOffsetEncoding.scale != 0.0f) {
                    float inv_scale = 1.0f / qp.scaleOffsetEncoding.scale;
                    int32_t offset = qp.scaleOffsetEncoding.offset;
                    for (size_t e = 0; e < rd; e++) {
                        int32_t q = (int32_t)lrintf(floatBuf[e] * inv_scale) - offset;
                        if (q < -32768) q = -32768;
                        else if (q > 32767) q = 32767;
                        p[e] = (int16_t)q;
                    }
                } else {
                    for (size_t e = 0; e < rd; e++) p[e] = (int16_t)floatBuf[e];
                }
            } else if (dt == QNN_DATATYPE_SFIXED_POINT_8) {  /* 0x0308 */
                int8_t* p = (int8_t*)dst;
                for (size_t e = 0; e < rd; e++) p[e] = (int8_t)floatBuf[e];
            } else {
                /* Fallback: raw copy */
                size_t copySize = slot->inputBufSizes[target_idx];
                if (rd * 4 < copySize) copySize = rd * 4;
                memcpy(dst, floatBuf, copySize);
                fprintf(stderr, "WARN: unknown input dtype 0x%04x for input[%d], raw copy\n",
                        (unsigned)dt, target_idx);
            }
            free(floatBuf);
            } /* end of non-FLOAT_32 else block */
        }

        idx++;
        tok = strtok_r(NULL, " \t", &saveptr);
    }

    return 0;
}

/* Write output tensors to result_dir, converting native->float32 */
static int write_outputs(ContextSlot* slot, const char* result_dir) {
    mkdirs(result_dir);

    for (uint32_t i = 0; i < slot->numOutputs; ++i) {
        Qnn_Tensor_t* t = &slot->outputs[i];
        Qnn_DataType_t dt = (t->version == QNN_TENSOR_VERSION_2)
            ? t->v2.dataType : t->v1.dataType;
        uint32_t rank = (t->version == QNN_TENSOR_VERSION_2)
            ? t->v2.rank : t->v1.rank;
        const uint32_t* dims = (t->version == QNN_TENSOR_VERSION_2)
            ? t->v2.dimensions : t->v1.dimensions;

        size_t numElems = 1;
        for (uint32_t d = 0; d < rank; d++) numElems *= dims[d];

        char out_path[MAX_PATH_LEN];
        snprintf(out_path, sizeof(out_path), "%s/%s.raw",
                 result_dir, slot->outputNames[i]);

        if (dt == QNN_DATATYPE_FLOAT_32) {  /* 0x0232 */
            if (write_raw_file(out_path, slot->outputBufs[i], slot->outputBufSizes[i]) != 0) {
                fprintf(stderr, "ERR: write_output_failed %s\n", out_path);
                return -1;
            }
        } else {
            float* f32 = (float*)malloc(numElems * sizeof(float));
            if (!f32) { fprintf(stderr, "ERR: malloc_output_convert\n"); return -1; }

            if (dt == QNN_DATATYPE_FLOAT_16) {  /* 0x0216 - IEEE half -> float32 */
                uint16_t* src = (uint16_t*)slot->outputBufs[i];
                for (size_t e = 0; e < numElems; e++) {
                    uint16_t h = src[e];
                    uint32_t sign = (h >> 15) & 1;
                    uint32_t exp  = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;
                    uint32_t f;
                    if (exp == 0) {
                        if (mant == 0) f = sign << 31;
                        else {
                            exp = 1;
                            while (!(mant & 0x400)) { mant <<= 1; exp--; }
                            mant &= 0x3FF;
                            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                        }
                    } else if (exp == 31) {
                        f = (sign << 31) | 0x7F800000 | (mant << 13);
                    } else {
                        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                    }
                    memcpy(&f32[e], &f, 4);
                }
            } else if (dt == QNN_DATATYPE_UFIXED_POINT_16) {  /* 0x0416 */
                Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
                    ? t->v2.quantizeParams : t->v1.quantizeParams;
                uint16_t* src = (uint16_t*)slot->outputBufs[i];
                if (qp.encodingDefinition == QNN_DEFINITION_DEFINED &&
                    qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                    float scale = qp.scaleOffsetEncoding.scale;
                    int32_t offset = qp.scaleOffsetEncoding.offset;
                    for (size_t e = 0; e < numElems; e++)
                        f32[e] = ((float)src[e] + (float)offset) * scale;
                } else {
                    for (size_t e = 0; e < numElems; e++) f32[e] = (float)src[e];
                }
            } else if (dt == QNN_DATATYPE_SFIXED_POINT_16) {  /* 0x0316 */
                Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
                    ? t->v2.quantizeParams : t->v1.quantizeParams;
                int16_t* src = (int16_t*)slot->outputBufs[i];
                if (qp.encodingDefinition == QNN_DEFINITION_DEFINED &&
                    qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                    float scale = qp.scaleOffsetEncoding.scale;
                    int32_t offset = qp.scaleOffsetEncoding.offset;
                    for (size_t e = 0; e < numElems; e++)
                        f32[e] = ((float)src[e] + (float)offset) * scale;
                } else {
                    for (size_t e = 0; e < numElems; e++) f32[e] = (float)src[e];
                }
            } else if (dt == QNN_DATATYPE_UFIXED_POINT_8) {  /* 0x0408 */
                Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
                    ? t->v2.quantizeParams : t->v1.quantizeParams;
                uint8_t* src = (uint8_t*)slot->outputBufs[i];
                if (qp.encodingDefinition == QNN_DEFINITION_DEFINED &&
                    qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
                    float scale = qp.scaleOffsetEncoding.scale;
                    int32_t offset = qp.scaleOffsetEncoding.offset;
                    for (size_t e = 0; e < numElems; e++)
                        f32[e] = ((float)src[e] + (float)offset) * scale;
                } else {
                    for (size_t e = 0; e < numElems; e++) f32[e] = (float)src[e];
                }
            } else if (dt == QNN_DATATYPE_INT_32) {
                int32_t* src = (int32_t*)slot->outputBufs[i];
                for (size_t e = 0; e < numElems; e++) f32[e] = (float)src[e];
            } else if (dt == QNN_DATATYPE_INT_16) {
                int16_t* src = (int16_t*)slot->outputBufs[i];
                for (size_t e = 0; e < numElems; e++) f32[e] = (float)src[e];
            } else if (dt == QNN_DATATYPE_INT_8) {
                int8_t* src = (int8_t*)slot->outputBufs[i];
                for (size_t e = 0; e < numElems; e++) f32[e] = (float)src[e];
            } else if (dt == QNN_DATATYPE_UINT_32) {
                uint32_t* src = (uint32_t*)slot->outputBufs[i];
                for (size_t e = 0; e < numElems; e++) f32[e] = (float)src[e];
            } else if (dt == QNN_DATATYPE_UINT_16) {
                uint16_t* src = (uint16_t*)slot->outputBufs[i];
                for (size_t e = 0; e < numElems; e++) f32[e] = (float)src[e];
            } else if (dt == QNN_DATATYPE_UINT_8) {
                uint8_t* src = (uint8_t*)slot->outputBufs[i];
                for (size_t e = 0; e < numElems; e++) f32[e] = (float)src[e];
            } else {
                memcpy(f32, slot->outputBufs[i],
                       numElems * 4 < slot->outputBufSizes[i] ?
                       numElems * 4 : slot->outputBufSizes[i]);
                fprintf(stderr, "WARN: unknown output dtype 0x%04x for output[%u], raw copy\n",
                        (unsigned)dt, i);
            }
            if (write_raw_file(out_path, f32, numElems * sizeof(float)) != 0) {
                free(f32);
                fprintf(stderr, "ERR: write_output_failed %s\n", out_path);
                return -1;
            }
            free(f32);
        }
    }
    return 0;
}

#define MAX_BATCH_LINES 8

static int cmd_run(const char* id, const char* input_list_path, const char* output_dir) {
    int si = find_slot(id);
    if (si < 0) {
        printf("ERR context_not_found %s\n", id);
        fflush(stdout);
        return -1;
    }
    ContextSlot* slot = &g_slots[si];

    /* Read all input-list lines (batch mode) */
    FILE* f = fopen(input_list_path, "r");
    if (!f) {
        printf("ERR cannot_open_input_list %s\n", input_list_path);
        fflush(stdout);
        return -1;
    }
    char batch_lines[MAX_BATCH_LINES][MAX_LINE_LEN];
    int num_batches = 0;
    while (fgets(batch_lines[num_batches], MAX_LINE_LEN, f)) {
        char* ln = batch_lines[num_batches];
        size_t len = strlen(ln);
        while (len > 0 && (ln[len - 1] == '\n' || ln[len - 1] == '\r'))
            ln[--len] = '\0';
        if (len == 0 || ln[0] == '#') continue;
        num_batches++;
        if (num_batches >= MAX_BATCH_LINES) break;
    }
    fclose(f);

    if (num_batches == 0) {
        printf("ERR empty_input_list\n");
        fflush(stdout);
        return -1;
    }

    fprintf(stderr, "[server] RUN %s: %d batch(es) from %s\n", id, num_batches, input_list_path);

    double total_exec_ms = 0;

    for (int b = 0; b < num_batches; b++) {
        /* Parse inputs for this batch */
        if (parse_input_line(batch_lines[b], slot) != 0) {
            printf("ERR input_parse_failed batch=%d\n", b);
            fflush(stdout);
            return -1;
        }

        /* Debug dump (first batch only) */
        if (b == 0) {
            fprintf(stderr, "[server] RUN %s: numIn=%u numOut=%u\n", id, slot->numInputs, slot->numOutputs);
            for (uint32_t i = 0; i < slot->numInputs; ++i) {
                Qnn_Tensor_t* t = &slot->inputs[i];
                if (t->version == QNN_TENSOR_VERSION_2) {
                    fprintf(stderr, "[dbg] in[%u] v2: id=%u dtype=%u memType=%u rank=%u bufSz=%u\n",
                            i, t->v2.id, t->v2.dataType, t->v2.memType, t->v2.rank, t->v2.clientBuf.dataSize);
                }
            }
            for (uint32_t i = 0; i < slot->numOutputs; ++i) {
                Qnn_Tensor_t* t = &slot->outputs[i];
                if (t->version == QNN_TENSOR_VERSION_2) {
                    fprintf(stderr, "[dbg] out[%u] v2: id=%u dtype=%u memType=%u rank=%u bufSz=%u\n",
                            i, t->v2.id, t->v2.dataType, t->v2.memType, t->v2.rank, t->v2.clientBuf.dataSize);
                }
            }
        }

        /* Execute graph */
        double t0 = now_ms();
        Qnn_ErrorHandle_t err = g_qnn.graphExecute(
            slot->graphHandle,
            slot->inputs, slot->numInputs,
            slot->outputs, slot->numOutputs,
            NULL, NULL);
        double t1 = now_ms();

        if (QNN_SUCCESS != err) {
            printf("ERR graphExecute_failed %d batch=%d\n", (int)err, b);
            fflush(stdout);
            return -1;
        }

        double exec_ms = t1 - t0;
        total_exec_ms += exec_ms;
        fprintf(stderr, "[server] RUN %s batch[%d]: execute=%.1fms\n", id, b, exec_ms);

        /* Write outputs to Result_{b}/ */
        char result_dir[MAX_PATH_LEN];
        snprintf(result_dir, sizeof(result_dir), "%s/Result_%d", output_dir, b);

        if (write_outputs(slot, result_dir) != 0) {
            printf("ERR write_outputs_failed batch=%d\n", b);
            fflush(stdout);
            return -1;
        }
    }

    fprintf(stderr, "[server] RUN %s: total=%.1fms (%d batch)\n", id, total_exec_ms, num_batches);
    printf("OK %.1f\n", total_exec_ms);
    fflush(stdout);
    return 0;
}

/* ========================================================================= */
/*  RUN_CHAIN: encoder→decoder in memory (no intermediate file I/O)         */
/* ========================================================================= */

/*
 * RUN_CHAIN <enc_id> <dec_id> <enc_input_list> <dec_input_list> <output_dir> [enc_out:dec_in ...]
 *
 * For each batch line:
 *   1. Parse encoder inputs from enc_input_list
 *   2. graphExecute encoder
 *   3. Copy encoder output buffers → decoder input buffers per mapping
 *   4. Parse decoder inputs from dec_input_list (only unmapped inputs)
 *   5. graphExecute decoder
 *   6. Write decoder outputs to output_dir/Result_{batch_idx}/
 *
 * Eliminates ~214MB of disk I/O per batch for SDXL split UNet.
 */
static int cmd_run_chain(const char* enc_id, const char* dec_id,
                          const char* enc_il_path, const char* dec_il_path,
                          const char* output_dir,
                          int argc, char** argv) {
    int enc_si = find_slot(enc_id);
    int dec_si = find_slot(dec_id);
    if (enc_si < 0) { printf("ERR enc_context_not_found %s\n", enc_id); fflush(stdout); return -1; }
    if (dec_si < 0) { printf("ERR dec_context_not_found %s\n", dec_id); fflush(stdout); return -1; }
    ContextSlot* enc = &g_slots[enc_si];
    ContextSlot* dec = &g_slots[dec_si];

    /* Parse mappings: enc_output_name:dec_input_name */
    typedef struct { int enc_out_idx; int dec_in_idx; size_t copy_size; } PipeMap;
    PipeMap pipes[MAX_TENSORS];
    int num_pipes = 0;
    int dec_piped[MAX_TENSORS];
    memset(dec_piped, 0, sizeof(dec_piped));

    for (int a = 0; a < argc; a++) {
        char* colon = strchr(argv[a], ':');
        if (!colon) continue;
        *colon = '\0';
        const char* enc_out_name = argv[a];
        const char* dec_in_name = colon + 1;

        int eidx = -1, didx = -1;
        for (uint32_t j = 0; j < enc->numOutputs; j++) {
            if (strcmp(enc->outputNames[j], enc_out_name) == 0) { eidx = (int)j; break; }
        }
        for (uint32_t j = 0; j < dec->numInputs; j++) {
            if (strcmp(dec->inputNames[j], dec_in_name) == 0) { didx = (int)j; break; }
        }
        if (eidx < 0 || didx < 0) {
            fprintf(stderr, "[server] WARN: chain mapping '%s:%s' not resolved (enc_out=%d dec_in=%d)\n",
                    enc_out_name, dec_in_name, eidx, didx);
            *colon = ':'; /* restore */
            continue;
        }
        /* Verify buffer sizes match */
        if (enc->outputBufSizes[eidx] != dec->inputBufSizes[didx]) {
            fprintf(stderr, "[server] ERR: chain pipe size mismatch: enc out[%d]=%zu dec in[%d]=%zu\n",
                    eidx, enc->outputBufSizes[eidx], didx, dec->inputBufSizes[didx]);
            printf("ERR chain_pipe_size_mismatch %s:%s\n", enc_out_name, dec_in_name);
            fflush(stdout);
            *colon = ':';
            return -1;
        }
        pipes[num_pipes].enc_out_idx = eidx;
        pipes[num_pipes].dec_in_idx = didx;
        pipes[num_pipes].copy_size = enc->outputBufSizes[eidx];
        num_pipes++;
        dec_piped[didx] = 1;
        *colon = ':';
    }

    fprintf(stderr, "[server] RUN_CHAIN enc=%s dec=%s pipes=%d\n", enc_id, dec_id, num_pipes);

    /* Read encoder input-list lines */
    FILE* f_enc = fopen(enc_il_path, "r");
    if (!f_enc) { printf("ERR cannot_open_enc_input_list %s\n", enc_il_path); fflush(stdout); return -1; }
    char enc_lines[MAX_BATCH_LINES][MAX_LINE_LEN];
    int num_batches = 0;
    while (fgets(enc_lines[num_batches], MAX_LINE_LEN, f_enc)) {
        char* ln = enc_lines[num_batches];
        size_t len = strlen(ln);
        while (len > 0 && (ln[len - 1] == '\n' || ln[len - 1] == '\r')) ln[--len] = '\0';
        if (len == 0 || ln[0] == '#') continue;
        num_batches++;
        if (num_batches >= MAX_BATCH_LINES) break;
    }
    fclose(f_enc);

    /* Read decoder input-list lines */
    FILE* f_dec = fopen(dec_il_path, "r");
    if (!f_dec) { printf("ERR cannot_open_dec_input_list %s\n", dec_il_path); fflush(stdout); return -1; }
    char dec_lines[MAX_BATCH_LINES][MAX_LINE_LEN];
    int num_dec_lines = 0;
    while (fgets(dec_lines[num_dec_lines], MAX_LINE_LEN, f_dec)) {
        char* ln = dec_lines[num_dec_lines];
        size_t len = strlen(ln);
        while (len > 0 && (ln[len - 1] == '\n' || ln[len - 1] == '\r')) ln[--len] = '\0';
        if (len == 0 || ln[0] == '#') continue;
        num_dec_lines++;
        if (num_dec_lines >= MAX_BATCH_LINES) break;
    }
    fclose(f_dec);

    if (num_batches == 0) { printf("ERR empty_enc_input_list\n"); fflush(stdout); return -1; }
    if (num_dec_lines != num_batches) {
        fprintf(stderr, "[server] WARN: batch count mismatch enc=%d dec=%d, using min\n",
                num_batches, num_dec_lines);
        if (num_dec_lines < num_batches) num_batches = num_dec_lines;
    }

    double total_enc_ms = 0, total_dec_ms = 0;

    for (int b = 0; b < num_batches; b++) {
        /* 1. Parse encoder inputs */
        if (parse_input_line(enc_lines[b], enc) != 0) {
            printf("ERR enc_input_parse_failed batch=%d\n", b);
            fflush(stdout);
            return -1;
        }

        /* 2. Execute encoder */
        double t0 = now_ms();
        Qnn_ErrorHandle_t err = g_qnn.graphExecute(
            enc->graphHandle, enc->inputs, enc->numInputs,
            enc->outputs, enc->numOutputs, NULL, NULL);
        double t1 = now_ms();
        if (QNN_SUCCESS != err) {
            printf("ERR enc_graphExecute_failed %d batch=%d\n", (int)err, b);
            fflush(stdout);
            return -1;
        }
        total_enc_ms += t1 - t0;

        /* 3. Pipe encoder outputs → decoder inputs (memcpy) */
        for (int p = 0; p < num_pipes; p++) {
            memcpy(dec->inputBufs[pipes[p].dec_in_idx],
                   enc->outputBufs[pipes[p].enc_out_idx],
                   pipes[p].copy_size);
        }

        /* 4. Parse decoder non-piped inputs */
        if (dec_il_path[0] != '\0') {
            /* Only read non-piped inputs; piped data stays in buffer */
            if (parse_input_line(dec_lines[b], dec) != 0) {
                printf("ERR dec_input_parse_failed batch=%d\n", b);
                fflush(stdout);
                return -1;
            }
        }

        /* 5. Execute decoder */
        double t2 = now_ms();
        err = g_qnn.graphExecute(
            dec->graphHandle, dec->inputs, dec->numInputs,
            dec->outputs, dec->numOutputs, NULL, NULL);
        double t3 = now_ms();
        if (QNN_SUCCESS != err) {
            printf("ERR dec_graphExecute_failed %d batch=%d\n", (int)err, b);
            fflush(stdout);
            return -1;
        }
        total_dec_ms += t3 - t2;

        /* 6. Write decoder outputs */
        char result_dir[MAX_PATH_LEN];
        snprintf(result_dir, sizeof(result_dir), "%s/Result_%d", output_dir, b);
        if (write_outputs(dec, result_dir) != 0) {
            printf("ERR write_outputs_failed batch=%d\n", b);
            fflush(stdout);
            return -1;
        }

        fprintf(stderr, "[server] CHAIN batch[%d]: enc=%.1fms dec=%.1fms\n",
                b, t1 - t0, t3 - t2);
    }

    double total = total_enc_ms + total_dec_ms;
    fprintf(stderr, "[server] RUN_CHAIN: enc=%.1fms dec=%.1fms total=%.1fms (%d batch)\n",
            total_enc_ms, total_dec_ms, total, num_batches);
    printf("OK %.1f\n", total);
    fflush(stdout);
    return 0;
}

/* ========================================================================= */
/*  DENOISE: Autonomous 8-Step Denoise Loop in Server Memory (Zero-Copy)    */
/* ========================================================================= */

typedef struct {
    int   step;
    float timestep;
    float sigma;
    float sigma_next;
} DenoiseScheduleStep;

static inline float fp16_to_f32(uint16_t h) {
#if defined(__aarch64__)
    __fp16 fh;
    memcpy(&fh, &h, 2);
    return (float)fh;
#else
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float res;
    memcpy(&res, &f, 4);
    return res;
#endif
}

static inline uint16_t f32_to_fp16(float val) {
#if defined(__aarch64__)
    __fp16 fh = (__fp16)val;
    uint16_t h;
    memcpy(&h, &fh, 2);
    return h;
#else
    uint32_t bits;
    memcpy(&bits, &val, 4);
    uint32_t s = (bits >> 16) & 0x8000;
    int32_t ex = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t m = bits & 0x007FFFFF;
    if (ex <= 0) return (uint16_t)s;
    if (ex >= 31) return (uint16_t)(s | 0x7C00);
    return (uint16_t)(s | (ex << 10) | (m >> 13));
#endif
}

static void tensor_get_f32(ContextSlot* slot, uint32_t idx, float* dst, size_t numElems) {
    Qnn_Tensor_t* t = &slot->outputs[idx];
    Qnn_DataType_t dt = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.dataType : t->v1.dataType;
    void* src = slot->outputBufs[idx];

    if (dt == QNN_DATATYPE_FLOAT_32) {
        memcpy(dst, src, numElems * sizeof(float));
    } else if (dt == QNN_DATATYPE_FLOAT_16) {
        const uint16_t* s = (const uint16_t*)src;
        for (size_t e = 0; e < numElems; e++) {
            dst[e] = fp16_to_f32(s[e]);
        }
    } else if (dt == QNN_DATATYPE_UFIXED_POINT_16) {
        Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
            ? t->v2.quantizeParams : t->v1.quantizeParams;
        const uint16_t* s = (const uint16_t*)src;
        if (qp.encodingDefinition == QNN_DEFINITION_DEFINED &&
            qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
            float scale = qp.scaleOffsetEncoding.scale;
            int32_t offset = qp.scaleOffsetEncoding.offset;
            for (size_t e = 0; e < numElems; e++) {
                dst[e] = ((float)s[e] + (float)offset) * scale;
            }
        } else {
            for (size_t e = 0; e < numElems; e++) dst[e] = (float)s[e];
        }
    } else if (dt == QNN_DATATYPE_SFIXED_POINT_16) {
        Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
            ? t->v2.quantizeParams : t->v1.quantizeParams;
        const int16_t* s = (const int16_t*)src;
        if (qp.encodingDefinition == QNN_DEFINITION_DEFINED &&
            qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
            float scale = qp.scaleOffsetEncoding.scale;
            int32_t offset = qp.scaleOffsetEncoding.offset;
            for (size_t e = 0; e < numElems; e++) {
                dst[e] = ((float)s[e] + (float)offset) * scale;
            }
        } else {
            for (size_t e = 0; e < numElems; e++) dst[e] = (float)s[e];
        }
    } else if (dt == QNN_DATATYPE_UFIXED_POINT_8) {
        Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
            ? t->v2.quantizeParams : t->v1.quantizeParams;
        const uint8_t* s = (const uint8_t*)src;
        if (qp.encodingDefinition == QNN_DEFINITION_DEFINED &&
            qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
            float scale = qp.scaleOffsetEncoding.scale;
            int32_t offset = qp.scaleOffsetEncoding.offset;
            for (size_t e = 0; e < numElems; e++) {
                dst[e] = ((float)s[e] + (float)offset) * scale;
            }
        } else {
            for (size_t e = 0; e < numElems; e++) dst[e] = (float)s[e];
        }
    } else if (dt == QNN_DATATYPE_INT_32) {
        const int32_t* s = (const int32_t*)src;
        for (size_t e = 0; e < numElems; e++) dst[e] = (float)s[e];
    } else {
        memcpy(dst, src, numElems * sizeof(float));
    }
}

static void tensor_set_f32(ContextSlot* slot, uint32_t idx, const float* src, size_t numElems) {
    Qnn_Tensor_t* t = &slot->inputs[idx];
    Qnn_DataType_t dt = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.dataType : t->v1.dataType;
    void* dst = slot->inputBufs[idx];

    if (dt == QNN_DATATYPE_FLOAT_32) {
        memcpy(dst, src, numElems * sizeof(float));
    } else if (dt == QNN_DATATYPE_FLOAT_16) {
        uint16_t* p = (uint16_t*)dst;
        for (size_t e = 0; e < numElems; e++) {
            p[e] = f32_to_fp16(src[e]);
        }
    } else if (dt == QNN_DATATYPE_UFIXED_POINT_16) {
        Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
            ? t->v2.quantizeParams : t->v1.quantizeParams;
        uint16_t* p = (uint16_t*)dst;
        if (qp.encodingDefinition == QNN_DEFINITION_DEFINED &&
            qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET &&
            qp.scaleOffsetEncoding.scale != 0.0f) {
            float inv_scale = 1.0f / qp.scaleOffsetEncoding.scale;
            int32_t offset = qp.scaleOffsetEncoding.offset;
            for (size_t e = 0; e < numElems; e++) {
                int32_t q = (int32_t)lrintf(src[e] * inv_scale) - offset;
                if (q < 0) q = 0;
                else if (q > 65535) q = 65535;
                p[e] = (uint16_t)q;
            }
        } else {
            for (size_t e = 0; e < numElems; e++) p[e] = (uint16_t)src[e];
        }
    } else if (dt == QNN_DATATYPE_SFIXED_POINT_16) {
        Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
            ? t->v2.quantizeParams : t->v1.quantizeParams;
        int16_t* p = (int16_t*)dst;
        if (qp.encodingDefinition == QNN_DEFINITION_DEFINED &&
            qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET &&
            qp.scaleOffsetEncoding.scale != 0.0f) {
            float inv_scale = 1.0f / qp.scaleOffsetEncoding.scale;
            int32_t offset = qp.scaleOffsetEncoding.offset;
            for (size_t e = 0; e < numElems; e++) {
                int32_t q = (int32_t)lrintf(src[e] * inv_scale) - offset;
                if (q < -32768) q = -32768;
                else if (q > 32767) q = 32767;
                p[e] = (int16_t)q;
            }
        } else {
            for (size_t e = 0; e < numElems; e++) p[e] = (int16_t)src[e];
        }
    } else if (dt == QNN_DATATYPE_UFIXED_POINT_8) {
        Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
            ? t->v2.quantizeParams : t->v1.quantizeParams;
        uint8_t* p = (uint8_t*)dst;
        if (qp.encodingDefinition == QNN_DEFINITION_DEFINED &&
            qp.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET &&
            qp.scaleOffsetEncoding.scale != 0.0f) {
            float inv_scale = 1.0f / qp.scaleOffsetEncoding.scale;
            int32_t offset = qp.scaleOffsetEncoding.offset;
            for (size_t e = 0; e < numElems; e++) {
                int32_t q = (int32_t)lrintf(src[e] * inv_scale) - offset;
                if (q < 0) q = 0;
                else if (q > 255) q = 255;
                p[e] = (uint8_t)q;
            }
        } else {
            for (size_t e = 0; e < numElems; e++) p[e] = (uint8_t)src[e];
        }
    } else if (dt == QNN_DATATYPE_INT_32) {
        int32_t* p = (int32_t*)dst;
        for (size_t e = 0; e < numElems; e++) p[e] = (int32_t)src[e];
    } else {
        memcpy(dst, src, numElems * datatype_size(dt));
    }
}

/* ========================================================================= */
/*  Monolithic UNet Time/Aug Embedding + 17 ResNet Bias Engine (utemb)       */
/* ========================================================================= */

typedef struct {
    uint32_t  out_dim;
    uint32_t  in_dim;
    uint16_t* w_fp16;
    float*    b_fp32;
} TembLinearLayer;

typedef struct {
    int             loaded;
    void*           raw_blob;
    size_t          raw_size;
    TembLinearLayer layers[21]; /* 0..1: time_emb, 2..3: add_emb, 4..20: 17 resnets */
} UnetTembModel;

static UnetTembModel g_temb = {0};

static int ensure_temb_loaded(const char* context_binary_path) {
    if (g_temb.loaded) return 0;

    char candidates[4][MAX_PATH_LEN];
    int num_cand = 0;

    const char* env_p = getenv("SDXL_TEMB_BIN");
    if (env_p && env_p[0]) {
        strncpy(candidates[num_cand++], env_p, MAX_PATH_LEN - 1);
    }
    if (context_binary_path && context_binary_path[0]) {
        char dir[MAX_PATH_LEN];
        strncpy(dir, context_binary_path, sizeof(dir) - 1);
        dir[sizeof(dir) - 1] = '\0';
        char* slash = strrchr(dir, '/');
        if (slash) {
            *slash = '\0';
            snprintf(candidates[num_cand++], MAX_PATH_LEN, "%s/unet_temb_fp16.bin", dir);
        }
    }
    snprintf(candidates[num_cand++], MAX_PATH_LEN, "/sdcard/Download/sdxl_qnn/context/unet_temb_fp16.bin");
    snprintf(candidates[num_cand++], MAX_PATH_LEN, "/data/local/tmp/sdxl_test/unet_temb_fp16.bin");

    void* blob = NULL;
    size_t sz = 0;
    const char* loaded_path = NULL;
    for (int i = 0; i < num_cand; ++i) {
        if (load_file_malloc(candidates[i], &blob, &sz) == 0 && sz > 16) {
            loaded_path = candidates[i];
            break;
        }
    }
    if (!blob) {
        fprintf(stderr, "[temb] ERR: could not find unet_temb_fp16.bin!\n");
        return -1;
    }

    const uint8_t* ptr = (const uint8_t*)blob;
    const uint8_t* end = ptr + sz;
    if (memcmp(ptr, "UTMB", 4) != 0) {
        fprintf(stderr, "[temb] ERR: invalid magic in %s\n", loaded_path);
        free(blob);
        return -1;
    }
    uint32_t ver = 0;
    memcpy(&ver, ptr + 4, 4);
    ptr += 8;

    for (int i = 0; i < 21; ++i) {
        if (ptr + 8 > end) { free(blob); return -1; }
        uint32_t out_d = 0, in_d = 0;
        memcpy(&out_d, ptr, 4);
        memcpy(&in_d, ptr + 4, 4);
        ptr += 8;
        size_t w_bytes = (size_t)out_d * (size_t)in_d * sizeof(uint16_t);
        size_t b_bytes = (size_t)out_d * sizeof(float);
        if (ptr + w_bytes + b_bytes > end) {
            fprintf(stderr, "[temb] ERR: truncated layer %d in %s\n", i, loaded_path);
            free(blob);
            return -1;
        }
        g_temb.layers[i].out_dim = out_d;
        g_temb.layers[i].in_dim  = in_d;
        g_temb.layers[i].w_fp16  = (uint16_t*)ptr;
        ptr += w_bytes;
        g_temb.layers[i].b_fp32  = (float*)ptr;
        ptr += b_bytes;
    }

    g_temb.raw_blob = blob;
    g_temb.raw_size = sz;
    g_temb.loaded   = 1;
    fprintf(stderr, "[temb] Loaded 21 projection layers from %s (%.1f MB)\n",
            loaded_path, (double)sz / (1024.0 * 1024.0));
    return 0;
}

static void time_proj_f32(float val, int num_channels, float* out) {
    int half = num_channels / 2;
    const float log_10000 = 9.210340371976184f;
    for (int i = 0; i < half; ++i) {
        float freq = expf(-log_10000 * (float)i / (float)half);
        float arg = val * freq;
        out[i]        = cosf(arg); /* flip_sin_to_cos=True */
        out[half + i] = sinf(arg);
    }
}

static void silu_inplace(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

/* ========================================================================= */
/*  Hardware & Host Profiling Telemetry                                      */
/* ========================================================================= */

typedef struct {
    int      enabled;
    int      detailed;
    uint32_t seen_events;
    int      hvx_threads;
    double   temb_proj_ms;
    double   rpcmem_bias_ms;
    double   io_quant_ms;
    double   qnn_wall_ms;
    double   qnn_exec_us;
    double   qnn_device_us;
    double   qnn_device_excl_wait_us;
    double   qnn_host_rpc_us;
    double   qnn_htp_rpc_us;
    double   qnn_wait_us;
    double   qnn_pre_us;
    double   qnn_post_us;
    uint64_t qnn_device_cycles;
    int      unet_passes;
    double   vae_wall_ms;
    double   vae_device_us;
    uint64_t vae_device_cycles;
} PerfTelemetry;

static PerfTelemetry       g_perf = {0};
static Qnn_ProfileHandle_t g_profHandle = NULL;
static int                 g_use_legacy_temb = 0;

static void record_qnn_profile_Node(QnnProfile_EventId_t evId, int is_vae) {
    QnnProfile_EventType_t ev_type = 0;
    QnnProfile_EventUnit_t ev_unit = 0;
    uint64_t               ev_val  = 0;
    int                    got_ev  = 0;

    if (g_qnn.profileGetEventData) {
        QnnProfile_EventData_t data;
        memset(&data, 0, sizeof(data));
        if (QNN_SUCCESS == g_qnn.profileGetEventData(evId, &data)) {
            ev_type = data.type;
            ev_unit = data.unit;
            ev_val  = data.value;
            got_ev  = 1;
        }
    }
    if (!got_ev && g_qnn.profileGetExtendedEventData) {
        QnnProfile_ExtendedEventData_t ext = QNN_PROFILE_EXTENDED_EVENT_DATA_INIT;
        if (QNN_SUCCESS == g_qnn.profileGetExtendedEventData(evId, &ext)) {
            ev_type = ext.v1.type;
            ev_unit = ext.v1.unit;
            ev_val  = ext.v1.value.uint64Value;
            got_ev  = 1;
        }
    }

    if (got_ev) {
        if (!is_vae) {
            if (ev_type == QNN_PROFILE_EVENTTYPE_EXECUTE && ev_unit == QNN_PROFILE_EVENTUNIT_MICROSEC) {
                g_perf.qnn_exec_us += (double)ev_val;
            } else if (ev_type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HOST_RPC_TIME_MICROSEC) {
                g_perf.qnn_host_rpc_us += (double)ev_val;
            } else if (ev_type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HTP_RPC_TIME_MICROSEC) {
                g_perf.qnn_htp_rpc_us += (double)ev_val;
            } else if (ev_type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC ||
                       (ev_type == QNN_PROFILE_EVENTTYPE_EXECUTE_DEVICE && ev_unit == QNN_PROFILE_EVENTUNIT_MICROSEC)) {
                g_perf.qnn_device_us += (double)ev_val;
            } else if (ev_type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_EXCL_WAIT_TIME_MICROSEC) {
                g_perf.qnn_device_excl_wait_us += (double)ev_val;
            } else if (ev_type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_CYCLE ||
                       (ev_type == QNN_PROFILE_EVENTTYPE_EXECUTE_DEVICE && ev_unit == QNN_PROFILE_EVENTUNIT_CYCLES)) {
                g_perf.qnn_device_cycles += ev_val;
            } else if (ev_type == QNN_PROFILE_EVENTTYPE_EXECUTE_QUEUE_WAIT ||
                       ev_type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_VTCM_ACQUIRE_TIME ||
                       ev_type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_RESOURCE_POWER_UP_TIME) {
                g_perf.qnn_wait_us += (double)ev_val;
            } else if (ev_type == QNN_PROFILE_EVENTTYPE_EXECUTE_PREPROCESS) {
                g_perf.qnn_pre_us += (double)ev_val;
            } else if (ev_type == QNN_PROFILE_EVENTTYPE_EXECUTE_POSTPROCESS) {
                g_perf.qnn_post_us += (double)ev_val;
            } else if (ev_type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_NUMBER_OF_HVX_THREADS && ev_val > 0) {
                g_perf.hvx_threads = (int)ev_val;
            }
        } else {
            if (ev_type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC ||
                ev_type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_EXCL_WAIT_TIME_MICROSEC ||
                (ev_type == QNN_PROFILE_EVENTTYPE_EXECUTE_DEVICE && ev_unit == QNN_PROFILE_EVENTUNIT_MICROSEC)) {
                if (g_perf.vae_device_us == 0.0 || ev_type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_EXCL_WAIT_TIME_MICROSEC) {
                    g_perf.vae_device_us = (double)ev_val;
                }
            } else if (ev_type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_CYCLE ||
                       (ev_type == QNN_PROFILE_EVENTTYPE_EXECUTE_DEVICE && ev_unit == QNN_PROFILE_EVENTUNIT_CYCLES)) {
                g_perf.vae_device_cycles += ev_val;
            }
        }
    }

    if (g_qnn.profileGetSubEvents) {
        const QnnProfile_EventId_t* subEvents = NULL;
        uint32_t numSub = 0;
        if (QNN_SUCCESS == g_qnn.profileGetSubEvents(evId, &subEvents, &numSub) && subEvents) {
            for (uint32_t i = 0; i < numSub; ++i) {
                record_qnn_profile_Node(subEvents[i], is_vae);
            }
        }
    }
}

static void collect_qnn_profile_events(Qnn_ProfileHandle_t prof, int is_vae) {
    if (!prof || !g_qnn.profileGetEvents) return;
    const QnnProfile_EventId_t* events = NULL;
    uint32_t numEvents = 0;
    if (QNN_SUCCESS == g_qnn.profileGetEvents(prof, &events, &numEvents) && events) {
        for (uint32_t i = 0; i < numEvents; ++i) {
            record_qnn_profile_Node(events[i], is_vae);
        }
    }
}

static void temb_linear_fwd(const TembLinearLayer* L, const float* x, float* y) {
    uint32_t out_d = L->out_dim;
    uint32_t in_d  = L->in_dim;
#if defined(__aarch64__)
    if (!g_use_legacy_temb) {
        for (uint32_t o = 0; o < out_d; ++o) {
            const uint16_t* w_row = L->w_fp16 + (size_t)o * in_d;
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            uint32_t i = 0;
            for (; i + 8 <= in_d; i += 8) {
                float16x8_t w16 = vld1q_f16((const __fp16*)(w_row + i));
                float32x4_t w_lo = vcvt_f32_f16(vget_low_f16(w16));
                float32x4_t w_hi = vcvt_high_f32_f16(w16);
                float32x4_t x_lo = vld1q_f32(x + i);
                float32x4_t x_hi = vld1q_f32(x + i + 4);
                acc0 = vfmaq_f32(acc0, w_lo, x_lo);
                acc1 = vfmaq_f32(acc1, w_hi, x_hi);
            }
            float sum = L->b_fp32[o] + vaddvq_f32(vaddq_f32(acc0, acc1));
            for (; i < in_d; ++i) {
                sum += fp16_to_f32(w_row[i]) * x[i];
            }
            y[o] = sum;
        }
        return;
    }
#endif
    for (uint32_t o = 0; o < out_d; ++o) {
        const uint16_t* w_row = L->w_fp16 + (size_t)o * in_d;
        float sum = L->b_fp32[o];
        for (uint32_t i = 0; i < in_d; ++i) {
            sum += fp16_to_f32(w_row[i]) * x[i];
        }
        y[o] = sum;
    }
}

typedef struct {
    int          valid;
    const float* te_ptr;
    float        te_head[4];
    float        tid[6];
    float        aug_emb[1280];
} AugEmbCacheSlot;

static AugEmbCacheSlot g_aug_cache[2] = {{0}};

static int compute_and_set_unet_resnet_biases(ContextSlot* slot, float timestep,
                                              const float* text_embeds, const float* time_ids) {
    if (ensure_temb_loaded(slot->binaryPath) != 0) return -1;

    double t0 = now_ms();
    float t_emb[320];
    time_proj_f32(timestep, 320, t_emb);

    float h1[1280], emb[1280], aug_emb[1280];
    temb_linear_fwd(&g_temb.layers[0], t_emb, h1);
    silu_inplace(h1, 1280);
    temb_linear_fwd(&g_temb.layers[1], h1, emb);

    int cache_hit = 0;
    if (!g_use_legacy_temb) {
        for (int c = 0; c < 2; ++c) {
            if (g_aug_cache[c].valid && g_aug_cache[c].te_ptr == text_embeds &&
                memcmp(g_aug_cache[c].te_head, text_embeds, 4 * sizeof(float)) == 0 &&
                memcmp(g_aug_cache[c].tid, time_ids, 6 * sizeof(float)) == 0) {
                memcpy(aug_emb, g_aug_cache[c].aug_emb, 1280 * sizeof(float));
                cache_hit = 1;
                break;
            }
        }
    }
    if (!cache_hit) {
        float add_in[2816];
        memcpy(add_in, text_embeds, 1280 * sizeof(float));
        for (int k = 0; k < 6; ++k) {
            time_proj_f32(time_ids[k], 256, add_in + 1280 + k * 256);
        }
        temb_linear_fwd(&g_temb.layers[2], add_in, h1);
        silu_inplace(h1, 1280);
        temb_linear_fwd(&g_temb.layers[3], h1, aug_emb);
        if (!g_use_legacy_temb) {
            int slot_idx = g_aug_cache[0].valid ? 1 : 0;
            if (g_aug_cache[0].valid && g_aug_cache[0].te_ptr == text_embeds) slot_idx = 0;
            g_aug_cache[slot_idx].valid = 1;
            g_aug_cache[slot_idx].te_ptr = text_embeds;
            memcpy(g_aug_cache[slot_idx].te_head, text_embeds, 4 * sizeof(float));
            memcpy(g_aug_cache[slot_idx].tid, time_ids, 6 * sizeof(float));
            memcpy(g_aug_cache[slot_idx].aug_emb, aug_emb, 1280 * sizeof(float));
        }
    }

    for (int i = 0; i < 1280; ++i) emb[i] += aug_emb[i];
    silu_inplace(emb, 1280);
    g_perf.temb_proj_ms += (now_ms() - t0);

    static uint8_t staging_block[65536] __attribute__((aligned(64)));

    for (int k = 0; k < 17; ++k) {
        double tl0 = now_ms();
        const TembLinearLayer* L = &g_temb.layers[4 + k];
        uint32_t C = L->out_dim;
        float bias_1d[1280];
        temb_linear_fwd(L, emb, bias_1d);
        g_perf.temb_proj_ms += (now_ms() - tl0);

        uint32_t idx = (uint32_t)(2 + k);
        if (idx >= slot->numInputs) break;

        Qnn_Tensor_t* t = &slot->inputs[idx];
        Qnn_DataType_t dt = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.dataType : t->v1.dataType;
        uint32_t rank = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.rank : t->v1.rank;
        const uint32_t* dims = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.dimensions : t->v1.dimensions;
        size_t total_elems = 1;
        for (uint32_t d = 0; d < rank; ++d) total_elems *= dims[d];
        size_t spatial = (C > 0) ? (total_elems / C) : 0;

        double tb0 = now_ms();
        if (dt == QNN_DATATYPE_UFIXED_POINT_16 || dt == QNN_DATATYPE_FLOAT_16) {
            uint16_t row_q[1280];
            if (dt == QNN_DATATYPE_UFIXED_POINT_16) {
                Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
                    ? t->v2.quantizeParams : t->v1.quantizeParams;
                float inv_scale = (qp.scaleOffsetEncoding.scale != 0.0f)
                    ? (1.0f / qp.scaleOffsetEncoding.scale) : 1.0f;
                int32_t offset = qp.scaleOffsetEncoding.offset;
                for (uint32_t c = 0; c < C; ++c) {
                    int32_t q = (int32_t)lrintf(bias_1d[c] * inv_scale) - offset;
                    if (q < 0) q = 0;
                    else if (q > 65535) q = 65535;
                    row_q[c] = (uint16_t)q;
                }
            } else {
                for (uint32_t c = 0; c < C; ++c) row_q[c] = f32_to_fp16(bias_1d[c]);
            }

            uint16_t* dst = (uint16_t*)slot->inputBufs[idx];
            if (rank == 4 && dims[3] == C) {
                /* NHWC [1, H, W, C]: replicate row_q across H*W spatial positions */
                size_t row_bytes = (size_t)C * sizeof(uint16_t);
                if (!g_use_legacy_temb && spatial >= 16 && row_bytes <= 8192) {
                    /* Build 64KB cached block via exponential doubling, then burst-write to ION RPCMEM */
                    size_t block_rows = sizeof(staging_block) / row_bytes;
                    if (block_rows > spatial) block_rows = spatial;
                    memcpy(staging_block, row_q, row_bytes);
                    size_t filled = 1;
                    while (filled < block_rows) {
                        size_t step_r = (filled * 2 <= block_rows) ? filled : (block_rows - filled);
                        memcpy(staging_block + filled * row_bytes, staging_block, step_r * row_bytes);
                        filled += step_r;
                    }
                    size_t block_bytes = block_rows * row_bytes;
                    uint8_t* dst_u8 = (uint8_t*)dst;
                    size_t pos = 0;
                    for (; pos + block_rows <= spatial; pos += block_rows) {
                        memcpy(dst_u8 + pos * row_bytes, staging_block, block_bytes);
                    }
                    if (pos < spatial) {
                        memcpy(dst_u8 + pos * row_bytes, staging_block, (spatial - pos) * row_bytes);
                    }
                } else {
                    for (size_t s = 0; s < spatial; ++s) {
                        memcpy(dst + s * C, row_q, row_bytes);
                    }
                }
            } else {
                /* NCHW [1, C, H, W] */
                for (uint32_t c = 0; c < C; ++c) {
                    uint16_t v = row_q[c];
                    uint16_t* ch_dst = dst + (size_t)c * spatial;
                    for (size_t s = 0; s < spatial; ++s) ch_dst[s] = v;
                }
            }
        } else if (dt == QNN_DATATYPE_FLOAT_32) {
            float* dst = (float*)slot->inputBufs[idx];
            if (rank == 4 && dims[3] == C) {
                for (size_t s = 0; s < spatial; ++s) {
                    memcpy(dst + s * C, bias_1d, (size_t)C * sizeof(float));
                }
            } else {
                for (uint32_t c = 0; c < C; ++c) {
                    float v = bias_1d[c];
                    float* ch_dst = dst + (size_t)c * spatial;
                    for (size_t s = 0; s < spatial; ++s) ch_dst[s] = v;
                }
            }
        }
        g_perf.rpcmem_bias_ms += (now_ms() - tb0);
    }
    return 0;
}

/* Symmetric mirror-reflection coordinate helper for Dynamic Resolution Sub-Canvas Isolation */
static inline int reflect_coord(int c, int limit) {
    if (limit <= 1) return 0;
    int period = 2 * limit - 2;
    int m = c % period;
    if (m < 0) m += period;
    return (m < limit) ? m : (period - m);
}

/* Query compiled spatial dimensions (H, W) of a 4D tensor (NHWC or NCHW) */
static void get_tensor_spatial_hw(const Qnn_Tensor_t* t, int* out_h, int* out_w) {
    uint32_t rank = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.rank : t->v1.rank;
    const uint32_t* dims = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.dimensions : t->v1.dimensions;
    if (rank == 4 && dims) {
        if (dims[3] == 3 || dims[3] == 4 || dims[3] >= 320) {
            /* NHWC [1, H, W, C] */
            *out_h = (int)dims[1];
            *out_w = (int)dims[2];
        } else {
            /* NCHW [1, C, H, W] */
            *out_h = (int)dims[2];
            *out_w = (int)dims[3];
        }
    }
}

/*
 * Sub-canvas layout-aware sample setter:
 * Places active [1, 4, act_H, act_W] into top-left of graph [1, graph_H, graph_W, 4]
 * and fills [act_H..graph_H, act_W..graph_W] using symmetric mirror-reflection (pad_mode=0),
 * periodic tiling (pad_mode=1), or zeros (pad_mode=2), fused with UFIXED_POINT_16 quantization.
 */
static void unet_set_sample_subrect_nchw(ContextSlot* slot, uint32_t idx, const float* nchw,
                                         int act_H, int act_W, int pad_mode) {
    double t0 = now_ms();
    Qnn_Tensor_t* t = &slot->inputs[idx];
    Qnn_DataType_t dt = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.dataType : t->v1.dataType;
    uint32_t rank = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.rank : t->v1.rank;
    const uint32_t* dims = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.dimensions : t->v1.dimensions;
    size_t act_hw = (size_t)act_H * (size_t)act_W;

    int graph_H = act_H, graph_W = act_W;
    get_tensor_spatial_hw(t, &graph_H, &graph_W);
    size_t graph_hw = (size_t)graph_H * (size_t)graph_W;

    if (rank == 4 && dims[3] == 4 && dt == QNN_DATATYPE_UFIXED_POINT_16 && !g_use_legacy_temb) {
        Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
            ? t->v2.quantizeParams : t->v1.quantizeParams;
        float inv_scale = (qp.scaleOffsetEncoding.scale != 0.0f)
            ? (1.0f / qp.scaleOffsetEncoding.scale) : 1.0f;
        int32_t offset = qp.scaleOffsetEncoding.offset;
        int32_t zero_q = -offset;
        if (zero_q < 0) zero_q = 0; else if (zero_q > 65535) zero_q = 65535;

        uint16_t* dst = (uint16_t*)slot->inputBufs[idx];
        const float* c0 = nchw + 0 * act_hw;
        const float* c1 = nchw + 1 * act_hw;
        const float* c2 = nchw + 2 * act_hw;
        const float* c3 = nchw + 3 * act_hw;

        if (act_H == graph_H && act_W == graph_W) {
            for (size_t p = 0; p < graph_hw; ++p) {
                int32_t q0 = (int32_t)lrintf(c0[p] * inv_scale) - offset;
                int32_t q1 = (int32_t)lrintf(c1[p] * inv_scale) - offset;
                int32_t q2 = (int32_t)lrintf(c2[p] * inv_scale) - offset;
                int32_t q3 = (int32_t)lrintf(c3[p] * inv_scale) - offset;
                dst[p * 4 + 0] = (uint16_t)(q0 < 0 ? 0 : (q0 > 65535 ? 65535 : q0));
                dst[p * 4 + 1] = (uint16_t)(q1 < 0 ? 0 : (q1 > 65535 ? 65535 : q1));
                dst[p * 4 + 2] = (uint16_t)(q2 < 0 ? 0 : (q2 > 65535 ? 65535 : q2));
                dst[p * 4 + 3] = (uint16_t)(q3 < 0 ? 0 : (q3 > 65535 ? 65535 : q3));
            }
        } else {
            for (int gy = 0; gy < graph_H; ++gy) {
                int sy = (gy < act_H) ? gy : ((pad_mode == 1) ? (gy % act_H) : reflect_coord(gy, act_H));
                for (int gx = 0; gx < graph_W; ++gx) {
                    size_t gp = ((size_t)gy * (size_t)graph_W + (size_t)gx) * 4;
                    if (pad_mode == 2 && (gy >= act_H || gx >= act_W)) {
                        dst[gp + 0] = (uint16_t)zero_q;
                        dst[gp + 1] = (uint16_t)zero_q;
                        dst[gp + 2] = (uint16_t)zero_q;
                        dst[gp + 3] = (uint16_t)zero_q;
                    } else {
                        int sx = (gx < act_W) ? gx : ((pad_mode == 1) ? (gx % act_W) : reflect_coord(gx, act_W));
                        size_t sp = (size_t)sy * (size_t)act_W + (size_t)sx;
                        int32_t q0 = (int32_t)lrintf(c0[sp] * inv_scale) - offset;
                        int32_t q1 = (int32_t)lrintf(c1[sp] * inv_scale) - offset;
                        int32_t q2 = (int32_t)lrintf(c2[sp] * inv_scale) - offset;
                        int32_t q3 = (int32_t)lrintf(c3[sp] * inv_scale) - offset;
                        dst[gp + 0] = (uint16_t)(q0 < 0 ? 0 : (q0 > 65535 ? 65535 : q0));
                        dst[gp + 1] = (uint16_t)(q1 < 0 ? 0 : (q1 > 65535 ? 65535 : q1));
                        dst[gp + 2] = (uint16_t)(q2 < 0 ? 0 : (q2 > 65535 ? 65535 : q2));
                        dst[gp + 3] = (uint16_t)(q3 < 0 ? 0 : (q3 > 65535 ? 65535 : q3));
                    }
                }
            }
        }
        g_perf.io_quant_ms += (now_ms() - t0);
        return;
    }

    /* Fallback path (FP16 / FP32 / NCHW / legacy) */
    size_t num_elems = 4 * graph_hw;
    if (rank == 4 && dims[3] == 4) {
        float* nhwc = (float*)malloc(num_elems * sizeof(float));
        for (int gy = 0; gy < graph_H; ++gy) {
            int sy = (gy < act_H) ? gy : ((pad_mode == 1) ? (gy % act_H) : reflect_coord(gy, act_H));
            for (int gx = 0; gx < graph_W; ++gx) {
                size_t gp = (size_t)gy * (size_t)graph_W + (size_t)gx;
                if (pad_mode == 2 && (gy >= act_H || gx >= act_W)) {
                    nhwc[gp * 4 + 0] = 0.0f;
                    nhwc[gp * 4 + 1] = 0.0f;
                    nhwc[gp * 4 + 2] = 0.0f;
                    nhwc[gp * 4 + 3] = 0.0f;
                } else {
                    int sx = (gx < act_W) ? gx : ((pad_mode == 1) ? (gx % act_W) : reflect_coord(gx, act_W));
                    size_t sp = (size_t)sy * (size_t)act_W + (size_t)sx;
                    nhwc[gp * 4 + 0] = nchw[0 * act_hw + sp];
                    nhwc[gp * 4 + 1] = nchw[1 * act_hw + sp];
                    nhwc[gp * 4 + 2] = nchw[2 * act_hw + sp];
                    nhwc[gp * 4 + 3] = nchw[3 * act_hw + sp];
                }
            }
        }
        tensor_set_f32(slot, idx, nhwc, num_elems);
        free(nhwc);
    } else {
        float* pad_nchw = (float*)malloc(num_elems * sizeof(float));
        for (int c = 0; c < 4; ++c) {
            for (int gy = 0; gy < graph_H; ++gy) {
                int sy = (gy < act_H) ? gy : ((pad_mode == 1) ? (gy % act_H) : reflect_coord(gy, act_H));
                for (int gx = 0; gx < graph_W; ++gx) {
                    size_t gp = (size_t)c * graph_hw + (size_t)gy * (size_t)graph_W + (size_t)gx;
                    if (pad_mode == 2 && (gy >= act_H || gx >= act_W)) {
                        pad_nchw[gp] = 0.0f;
                    } else {
                        int sx = (gx < act_W) ? gx : ((pad_mode == 1) ? (gx % act_W) : reflect_coord(gx, act_W));
                        pad_nchw[gp] = nchw[(size_t)c * act_hw + (size_t)sy * (size_t)act_W + (size_t)sx];
                    }
                }
            }
        }
        tensor_set_f32(slot, idx, pad_nchw, num_elems);
        free(pad_nchw);
    }
    g_perf.io_quant_ms += (now_ms() - t0);
}

static void unet_set_sample_nchw(ContextSlot* slot, uint32_t idx, const float* nchw, int H, int W) {
    unet_set_sample_subrect_nchw(slot, idx, nchw, H, W, 0);
}

static void unet_set_enc_hidden(ContextSlot* slot, uint32_t idx, const float* pe_77x2048) {
    double t0 = now_ms();
    Qnn_Tensor_t* t = &slot->inputs[idx];
    uint32_t rank = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.rank : t->v1.rank;
    const uint32_t* dims = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.dimensions : t->v1.dimensions;
    const size_t seq = 77, dim = 2048;

    static const float* last_pe_ptr = NULL;
    static ContextSlot* last_slot_ptr = NULL;
    static uint32_t last_idx = 999;
    if (!g_use_legacy_temb && last_slot_ptr == slot && last_idx == idx && last_pe_ptr == pe_77x2048) {
        /* Already loaded into this slot's input buffer! */
        return;
    }

    if (rank == 3 && dims[1] == 2048 && dims[2] == 77) {
        /* Transpose [1, 77, 2048] -> [1, 2048, 77] */
        float* nfc = (float*)malloc(seq * dim * sizeof(float));
        for (size_t s = 0; s < seq; ++s) {
            for (size_t c = 0; c < dim; ++c) {
                nfc[c * seq + s] = pe_77x2048[s * dim + c];
            }
        }
        tensor_set_f32(slot, idx, nfc, seq * dim);
        free(nfc);
    } else {
        tensor_set_f32(slot, idx, pe_77x2048, seq * dim);
    }
    last_slot_ptr = slot;
    last_idx = idx;
    last_pe_ptr = pe_77x2048;
    g_perf.io_quant_ms += (now_ms() - t0);
}

static void unet_get_noise_pred_subrect_nchw(ContextSlot* slot, uint32_t idx, float* dst_nchw, int act_H, int act_W) {
    double t0 = now_ms();
    Qnn_Tensor_t* t = &slot->outputs[idx];
    Qnn_DataType_t dt = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.dataType : t->v1.dataType;
    uint32_t rank = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.rank : t->v1.rank;
    const uint32_t* dims = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.dimensions : t->v1.dimensions;
    size_t act_hw = (size_t)act_H * (size_t)act_W;

    int graph_H = act_H, graph_W = act_W;
    get_tensor_spatial_hw(t, &graph_H, &graph_W);
    size_t graph_hw = (size_t)graph_H * (size_t)graph_W;

    if (rank == 4 && dims[3] == 4 && dt == QNN_DATATYPE_UFIXED_POINT_16 && !g_use_legacy_temb) {
        Qnn_QuantizeParams_t qp = (t->version == QNN_TENSOR_VERSION_2)
            ? t->v2.quantizeParams : t->v1.quantizeParams;
        float scale = qp.scaleOffsetEncoding.scale;
        float bias  = (float)qp.scaleOffsetEncoding.offset * scale;
        const uint16_t* src = (const uint16_t*)slot->outputBufs[idx];
        float* d0 = dst_nchw + 0 * act_hw;
        float* d1 = dst_nchw + 1 * act_hw;
        float* d2 = dst_nchw + 2 * act_hw;
        float* d3 = dst_nchw + 3 * act_hw;

        for (int y = 0; y < act_H; ++y) {
            const uint16_t* row = src + ((size_t)y * (size_t)graph_W) * 4;
            size_t dst_row = (size_t)y * (size_t)act_W;
            for (int x = 0; x < act_W; ++x) {
                d0[dst_row + x] = (float)row[x * 4 + 0] * scale + bias;
                d1[dst_row + x] = (float)row[x * 4 + 1] * scale + bias;
                d2[dst_row + x] = (float)row[x * 4 + 2] * scale + bias;
                d3[dst_row + x] = (float)row[x * 4 + 3] * scale + bias;
            }
        }
        g_perf.io_quant_ms += (now_ms() - t0);
        return;
    }

    size_t num_elems = 4 * graph_hw;
    float* full_buf = (float*)malloc(num_elems * sizeof(float));
    tensor_get_f32(slot, idx, full_buf, num_elems);
    if (rank == 4 && dims[3] == 4) {
        for (int y = 0; y < act_H; ++y) {
            for (int x = 0; x < act_W; ++x) {
                size_t gp = (size_t)y * (size_t)graph_W + (size_t)x;
                size_t dp = (size_t)y * (size_t)act_W + (size_t)x;
                dst_nchw[0 * act_hw + dp] = full_buf[gp * 4 + 0];
                dst_nchw[1 * act_hw + dp] = full_buf[gp * 4 + 1];
                dst_nchw[2 * act_hw + dp] = full_buf[gp * 4 + 2];
                dst_nchw[3 * act_hw + dp] = full_buf[gp * 4 + 3];
            }
        }
    } else {
        for (int c = 0; c < 4; ++c) {
            for (int y = 0; y < act_H; ++y) {
                memcpy(dst_nchw + (size_t)c * act_hw + (size_t)y * (size_t)act_W,
                       full_buf + (size_t)c * graph_hw + (size_t)y * (size_t)graph_W,
                       (size_t)act_W * sizeof(float));
            }
        }
    }
    free(full_buf);
    g_perf.io_quant_ms += (now_ms() - t0);
}

static void unet_get_noise_pred_nchw(ContextSlot* slot, uint32_t idx, float* dst_nchw, int H, int W) {
    unet_get_noise_pred_subrect_nchw(slot, idx, dst_nchw, H, W);
}

static void vae_get_rgb_subrect(ContextSlot* vae, uint32_t idx, float* dst_rgb,
                                int crop_y, int crop_x, int act_H, int act_W) {
    Qnn_Tensor_t* t = &vae->outputs[idx];
    int vae_H = act_H, vae_W = act_W;
    get_tensor_spatial_hw(t, &vae_H, &vae_W);
    size_t vae_elems = (size_t)vae_H * (size_t)vae_W * 3;

    if (crop_y == 0 && crop_x == 0 && act_H == vae_H && act_W == vae_W) {
        tensor_get_f32(vae, idx, dst_rgb, vae_elems);
        return;
    }
    float* full_rgb = (float*)malloc(vae_elems * sizeof(float));
    tensor_get_f32(vae, idx, full_rgb, vae_elems);
    for (int y = 0; y < act_H; ++y) {
        int sy = crop_y + y;
        if (sy < 0) sy = 0; else if (sy >= vae_H) sy = vae_H - 1;
        memcpy(dst_rgb + (size_t)y * (size_t)act_W * 3,
               full_rgb + ((size_t)sy * (size_t)vae_W + (size_t)crop_x) * 3,
               (size_t)act_W * 3 * sizeof(float));
    }
    free(full_rgb);
}

static int load_f32_raw(const char* path, float** out_buf, size_t* out_count) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz % sizeof(float) != 0) {
        fclose(f);
        return -2;
    }
    size_t count = (size_t)sz / sizeof(float);
    float* buf = (float*)malloc(sz);
    if (!buf) { fclose(f); return -3; }
    if (fread(buf, sizeof(float), count, f) != count) {
        free(buf);
        fclose(f);
        return -4;
    }
    fclose(f);
    *out_buf = buf;
    *out_count = count;
    return 0;
}

static int find_slot_input(ContextSlot* slot, const char* hint, uint32_t expected_elements) {
    for (uint32_t i = 0; i < slot->numInputs; ++i) {
        if (strstr(slot->inputNames[i], hint) != NULL) {
            return (int)i;
        }
    }
    if (expected_elements > 0) {
        for (uint32_t i = 0; i < slot->numInputs; ++i) {
            Qnn_Tensor_t* t = &slot->inputs[i];
            uint32_t rank = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.rank : t->v1.rank;
            const uint32_t* dims = (t->version == QNN_TENSOR_VERSION_2) ? t->v2.dimensions : t->v1.dimensions;
            size_t count = 1;
            for (uint32_t d = 0; d < rank; ++d) count *= dims[d];
            if (count == expected_elements) {
                return (int)i;
            }
        }
    }
    return -1;
}

static int cmd_denoise(const char* cfg_path) {
    FILE* fcfg = fopen(cfg_path, "r");
    if (!fcfg) {
        printf("ERR cannot_open_denoise_cfg %s\n", cfg_path);
        fflush(stdout);
        return -1;
    }

    char mode[32] = "chain";
    char enc_id[MAX_ID_LEN] = {0};
    char dec_id[MAX_ID_LEN] = {0};
    char init_latent_path[MAX_PATH_LEN] = {0};
    char out_latent_path[MAX_PATH_LEN] = {0};
    char schedule_path[MAX_PATH_LEN] = {0};
    char cond_dir[MAX_PATH_LEN] = {0};
    char uncond_dir[MAX_PATH_LEN] = {0};
    char preview_dir[MAX_PATH_LEN] = {0};
    float cfg_scale = 1.0f;
    int progressive_cfg = 999;
    int preview_stride = 0;
    int latent_h = 128;
    int latent_w = 128;
    char pipes_str[MAX_LINE_LEN] = {0};

    char line[MAX_LINE_LEN];
    while (fgets(line, sizeof(line), fcfg)) {
        char* ln = line;
        while (*ln == ' ' || *ln == '\t') ln++;
        size_t len = strlen(ln);
        while (len > 0 && (ln[len - 1] == '\n' || ln[len - 1] == '\r')) ln[--len] = '\0';
        if (len == 0 || ln[0] == '#') continue;

        char* eq = strchr(ln, '=');
        if (!eq) continue;
        *eq = '\0';
        char* key = ln;
        char* val = eq + 1;
        while (*val == ' ' || *val == '\t') val++;

        if (strcmp(key, "mode") == 0) strncpy(mode, val, sizeof(mode) - 1);
        else if (strcmp(key, "enc_id") == 0 || strcmp(key, "unet_id") == 0) strncpy(enc_id, val, sizeof(enc_id) - 1);
        else if (strcmp(key, "dec_id") == 0) strncpy(dec_id, val, sizeof(dec_id) - 1);
        else if (strcmp(key, "init_latent") == 0) strncpy(init_latent_path, val, sizeof(init_latent_path) - 1);
        else if (strcmp(key, "out_latent") == 0) strncpy(out_latent_path, val, sizeof(out_latent_path) - 1);
        else if (strcmp(key, "schedule_file") == 0) strncpy(schedule_path, val, sizeof(schedule_path) - 1);
        else if (strcmp(key, "cond_dir") == 0) strncpy(cond_dir, val, sizeof(cond_dir) - 1);
        else if (strcmp(key, "uncond_dir") == 0) strncpy(uncond_dir, val, sizeof(uncond_dir) - 1);
        else if (strcmp(key, "preview_dir") == 0) strncpy(preview_dir, val, sizeof(preview_dir) - 1);
        else if (strcmp(key, "cfg_scale") == 0) cfg_scale = (float)atof(val);
        else if (strcmp(key, "progressive_cfg") == 0) progressive_cfg = atoi(val);
        else if (strcmp(key, "preview_stride") == 0) preview_stride = atoi(val);
        else if (strcmp(key, "latent_h") == 0) latent_h = atoi(val);
        else if (strcmp(key, "latent_w") == 0) latent_w = atoi(val);
        else if (strcmp(key, "pipes") == 0) strncpy(pipes_str, val, sizeof(pipes_str) - 1);
    }
    fclose(fcfg);

    int is_chain = (strcmp(mode, "chain") == 0);
    int enc_si = find_slot(enc_id);
    if (enc_si < 0) {
        printf("ERR enc_or_unet_context_not_found %s\n", enc_id);
        fflush(stdout);
        return -1;
    }
    ContextSlot* enc = &g_slots[enc_si];
    ContextSlot* dec = NULL;

    typedef struct { int enc_out_idx; int dec_in_idx; size_t copy_size; } PipeMap;
    PipeMap pipes[MAX_TENSORS];
    int num_pipes = 0;

    if (is_chain) {
        int dec_si = find_slot(dec_id);
        if (dec_si < 0) {
            printf("ERR dec_context_not_found %s\n", dec_id);
            fflush(stdout);
            return -1;
        }
        dec = &g_slots[dec_si];

        char pstr_copy[MAX_LINE_LEN];
        strncpy(pstr_copy, pipes_str, sizeof(pstr_copy) - 1);
        char* ptok = strtok(pstr_copy, " \t");
        while (ptok && num_pipes < MAX_TENSORS) {
            char* col = strchr(ptok, ':');
            if (col) {
                *col = '\0';
                const char* e_name = ptok;
                const char* d_name = col + 1;
                int eidx = -1, didx = -1;
                for (uint32_t j = 0; j < enc->numOutputs; ++j) {
                    if (strcmp(enc->outputNames[j], e_name) == 0) { eidx = (int)j; break; }
                }
                for (uint32_t j = 0; j < dec->numInputs; ++j) {
                    if (strcmp(dec->inputNames[j], d_name) == 0) { didx = (int)j; break; }
                }
                if (eidx >= 0 && didx >= 0) {
                    pipes[num_pipes].enc_out_idx = eidx;
                    pipes[num_pipes].dec_in_idx = didx;
                    pipes[num_pipes].copy_size = (enc->outputBufSizes[eidx] < dec->inputBufSizes[didx])
                                                 ? enc->outputBufSizes[eidx] : dec->inputBufSizes[didx];
                    num_pipes++;
                }
            }
            ptok = strtok(NULL, " \t");
        }
    }

    FILE* fsched = fopen(schedule_path, "r");
    if (!fsched) {
        printf("ERR cannot_open_schedule %s\n", schedule_path);
        fflush(stdout);
        return -1;
    }
    DenoiseScheduleStep sched[64];
    int num_steps = 0;
    char sline[256];
    while (fgets(sline, sizeof(sline), fsched) && num_steps < 64) {
        int step = 0;
        float ts = 0.0f, s1 = 0.0f, s2 = 0.0f;
        if (sscanf(sline, "%d %f %f %f", &step, &ts, &s1, &s2) >= 4) {
            sched[num_steps].step = step;
            sched[num_steps].timestep = ts;
            sched[num_steps].sigma = s1;
            sched[num_steps].sigma_next = s2;
            num_steps++;
        }
    }
    fclose(fsched);
    if (num_steps == 0) {
        printf("ERR empty_schedule\n");
        fflush(stdout);
        return -1;
    }

    size_t num_latent = (size_t)1 * 4 * (size_t)latent_h * (size_t)latent_w;
    float* latent = NULL;
    size_t l_count = 0;
    if (load_f32_raw(init_latent_path, &latent, &l_count) != 0 || l_count != num_latent) {
        printf("ERR invalid_init_latent %s (expected %zu floats, got %zu)\n", init_latent_path, num_latent, l_count);
        fflush(stdout);
        if (latent) free(latent);
        return -1;
    }

    int is_ext_resnet = (!is_chain && enc->numInputs >= 19);
    int enc_smp_idx = find_slot_input(enc, "sample", (uint32_t)num_latent);
    int enc_ts_idx  = is_ext_resnet ? -1 : find_slot_input(enc, "timestep", 1);
    int enc_tid_idx = is_ext_resnet ? -1 : find_slot_input(enc, "time_ids", 6);
    int enc_te_idx  = is_ext_resnet ? -1 : find_slot_input(enc, "text_embeds", 1280);
    int enc_enc_idx = find_slot_input(enc, "encoder_hidden_states", 77 * 2048);

    int dec_enc_idx = -1;
    if (is_chain && dec) {
        dec_enc_idx = find_slot_input(dec, "encoder_hidden_states", 77 * 2048);
    }

    if (!is_ext_resnet) {
        if (enc_smp_idx < 0) enc_smp_idx = 4;
        if (enc_ts_idx < 0)  enc_ts_idx = 1;
        if (enc_tid_idx < 0) enc_tid_idx = 2;
        if (enc_te_idx < 0)  enc_te_idx = 3;
        if (enc_enc_idx < 0) enc_enc_idx = 0;
    } else {
        if (enc_smp_idx < 0) enc_smp_idx = 0;
        if (enc_enc_idx < 0) enc_enc_idx = 1;
    }
    if (dec && dec_enc_idx < 0) dec_enc_idx = 0;

    ContextSlot* final_out_slot = (is_chain && dec) ? dec : enc;
    int final_out_idx = 0;

    float *cond_enc = NULL, *cond_te = NULL, *cond_tid = NULL;
    size_t c_enc_cnt = 0, c_te_cnt = 0, c_tid_cnt = 0;
    char path_buf[MAX_PATH_LEN];

    snprintf(path_buf, sizeof(path_buf), "%s/enc.raw", cond_dir);
    load_f32_raw(path_buf, &cond_enc, &c_enc_cnt);
    snprintf(path_buf, sizeof(path_buf), "%s/te.raw", cond_dir);
    load_f32_raw(path_buf, &cond_te, &c_te_cnt);
    snprintf(path_buf, sizeof(path_buf), "%s/tid.raw", cond_dir);
    load_f32_raw(path_buf, &cond_tid, &c_tid_cnt);

    int has_uncond = (uncond_dir[0] != '\0' && strcmp(uncond_dir, "-") != 0 && cfg_scale > 1.0f);
    float *uncond_enc = NULL, *uncond_te = NULL, *uncond_tid = NULL;
    size_t u_enc_cnt = 0, u_te_cnt = 0, u_tid_cnt = 0;
    if (has_uncond) {
        snprintf(path_buf, sizeof(path_buf), "%s/enc.raw", uncond_dir);
        load_f32_raw(path_buf, &uncond_enc, &u_enc_cnt);
        snprintf(path_buf, sizeof(path_buf), "%s/te.raw", uncond_dir);
        load_f32_raw(path_buf, &uncond_te, &u_te_cnt);
        snprintf(path_buf, sizeof(path_buf), "%s/tid.raw", uncond_dir);
        load_f32_raw(path_buf, &uncond_tid, &u_tid_cnt);
    }

    float* scaled_sample = (float*)malloc(num_latent * sizeof(float));
    float* cond_pred_buf = (float*)malloc(num_latent * sizeof(float));
    float* uncond_pred_buf = has_uncond ? (float*)malloc(num_latent * sizeof(float)) : NULL;

    fprintf(stderr, "[server] DENOISE started: mode=%s (ext_resnet=%d) steps=%d cfg=%.2f progressive=%d latent=%dx%d\n",
            mode, is_ext_resnet, num_steps, cfg_scale, progressive_cfg, latent_h, latent_w);

    double t_start = now_ms();

    for (int s = 0; s < num_steps; ++s) {
        double t_step0 = now_ms();
        DenoiseScheduleStep step = sched[s];
        int step_uses_cfg = (has_uncond && s < progressive_cfg);

        float inv = 1.0f / sqrtf(step.sigma * step.sigma + 1.0f);
        for (size_t i = 0; i < num_latent; ++i) {
            scaled_sample[i] = latent[i] * inv;
        }
        unet_set_sample_nchw(enc, (uint32_t)enc_smp_idx, scaled_sample, latent_h, latent_w);
        if (enc_ts_idx >= 0) tensor_set_f32(enc, (uint32_t)enc_ts_idx, &step.timestep, 1);

        if (step_uses_cfg) {
            /* 1. Uncond pass */
            if (is_ext_resnet) {
                if (compute_and_set_unet_resnet_biases(enc, step.timestep, uncond_te, uncond_tid) != 0) {
                    printf("ERR temb_failed\n"); fflush(stdout); return -1;
                }
            } else {
                if (uncond_tid && enc_tid_idx >= 0) tensor_set_f32(enc, (uint32_t)enc_tid_idx, uncond_tid, u_tid_cnt);
                if (uncond_te && enc_te_idx >= 0)   tensor_set_f32(enc, (uint32_t)enc_te_idx, uncond_te, u_te_cnt);
            }
            if (uncond_enc) {
                if (enc_enc_idx >= 0 && (uint32_t)enc_enc_idx < enc->numInputs)
                    unet_set_enc_hidden(enc, (uint32_t)enc_enc_idx, uncond_enc);
                if (dec && dec_enc_idx >= 0 && (uint32_t)dec_enc_idx < dec->numInputs)
                    unet_set_enc_hidden(dec, (uint32_t)dec_enc_idx, uncond_enc);
            }
            g_qnn.graphExecute(enc->graphHandle, enc->inputs, enc->numInputs, enc->outputs, enc->numOutputs, NULL, NULL);
            if (is_chain && dec) {
                for (int p = 0; p < num_pipes; ++p) {
                    memcpy(dec->inputBufs[pipes[p].dec_in_idx], enc->outputBufs[pipes[p].enc_out_idx], pipes[p].copy_size);
                }
                g_qnn.graphExecute(dec->graphHandle, dec->inputs, dec->numInputs, dec->outputs, dec->numOutputs, NULL, NULL);
            }
            unet_get_noise_pred_nchw(final_out_slot, (uint32_t)final_out_idx, uncond_pred_buf, latent_h, latent_w);

            /* 2. Cond pass */
            if (is_ext_resnet) {
                compute_and_set_unet_resnet_biases(enc, step.timestep, cond_te, cond_tid);
            } else {
                if (cond_tid && enc_tid_idx >= 0) tensor_set_f32(enc, (uint32_t)enc_tid_idx, cond_tid, c_tid_cnt);
                if (cond_te && enc_te_idx >= 0)   tensor_set_f32(enc, (uint32_t)enc_te_idx, cond_te, c_te_cnt);
            }
            if (cond_enc) {
                if (enc_enc_idx >= 0 && (uint32_t)enc_enc_idx < enc->numInputs)
                    unet_set_enc_hidden(enc, (uint32_t)enc_enc_idx, cond_enc);
                if (dec && dec_enc_idx >= 0 && (uint32_t)dec_enc_idx < dec->numInputs)
                    unet_set_enc_hidden(dec, (uint32_t)dec_enc_idx, cond_enc);
            }
            g_qnn.graphExecute(enc->graphHandle, enc->inputs, enc->numInputs, enc->outputs, enc->numOutputs, NULL, NULL);
            if (is_chain && dec) {
                for (int p = 0; p < num_pipes; ++p) {
                    memcpy(dec->inputBufs[pipes[p].dec_in_idx], enc->outputBufs[pipes[p].enc_out_idx], pipes[p].copy_size);
                }
                g_qnn.graphExecute(dec->graphHandle, dec->inputs, dec->numInputs, dec->outputs, dec->numOutputs, NULL, NULL);
            }
            unet_get_noise_pred_nchw(final_out_slot, (uint32_t)final_out_idx, cond_pred_buf, latent_h, latent_w);

            /* CFG + Euler step in memory */
            float delta = step.sigma_next - step.sigma;
            for (size_t i = 0; i < num_latent; ++i) {
                float guided = uncond_pred_buf[i] + cfg_scale * (cond_pred_buf[i] - uncond_pred_buf[i]);
                latent[i] += delta * guided;
            }
        } else {
            /* Cond pass only */
            if (is_ext_resnet) {
                if (compute_and_set_unet_resnet_biases(enc, step.timestep, cond_te, cond_tid) != 0) {
                    printf("ERR temb_failed\n"); fflush(stdout); return -1;
                }
            } else {
                if (cond_tid && enc_tid_idx >= 0) tensor_set_f32(enc, (uint32_t)enc_tid_idx, cond_tid, c_tid_cnt);
                if (cond_te && enc_te_idx >= 0)   tensor_set_f32(enc, (uint32_t)enc_te_idx, cond_te, c_te_cnt);
            }
            if (cond_enc) {
                if (enc_enc_idx >= 0 && (uint32_t)enc_enc_idx < enc->numInputs)
                    unet_set_enc_hidden(enc, (uint32_t)enc_enc_idx, cond_enc);
                if (dec && dec_enc_idx >= 0 && (uint32_t)dec_enc_idx < dec->numInputs)
                    unet_set_enc_hidden(dec, (uint32_t)dec_enc_idx, cond_enc);
            }
            g_qnn.graphExecute(enc->graphHandle, enc->inputs, enc->numInputs, enc->outputs, enc->numOutputs, NULL, NULL);
            if (is_chain && dec) {
                for (int p = 0; p < num_pipes; ++p) {
                    memcpy(dec->inputBufs[pipes[p].dec_in_idx], enc->outputBufs[pipes[p].enc_out_idx], pipes[p].copy_size);
                }
                g_qnn.graphExecute(dec->graphHandle, dec->inputs, dec->numInputs, dec->outputs, dec->numOutputs, NULL, NULL);
            }
            unet_get_noise_pred_nchw(final_out_slot, (uint32_t)final_out_idx, cond_pred_buf, latent_h, latent_w);

            /* Euler step in memory */
            float delta = step.sigma_next - step.sigma;
            for (size_t i = 0; i < num_latent; ++i) {
                latent[i] += delta * cond_pred_buf[i];
            }
        }

        fprintf(stderr, "[server]   [UNet %d/%d]%s %.0fms\n",
                s + 1, num_steps, step_uses_cfg ? " CFG" : "", now_ms() - t_step0);

        /* Optional preview output */
        if (preview_stride > 0 && preview_dir[0] != '\0') {
            int is_last = (s == num_steps - 1);
            if (is_last || (s % preview_stride == preview_stride - 1)) {
                snprintf(path_buf, sizeof(path_buf), "%s/preview_step_%02d.raw", preview_dir, s + 1);
                write_raw_file(path_buf, latent, num_latent * sizeof(float));
            }
        }
    }

    double t_end = now_ms();
    double total_denoise_ms = t_end - t_start;

    /* Write final output latent */
    write_raw_file(out_latent_path, latent, num_latent * sizeof(float));

    /* Cleanup temporary memory */
    if (latent) free(latent);
    if (scaled_sample) free(scaled_sample);
    if (cond_pred_buf) free(cond_pred_buf);
    if (uncond_pred_buf) free(uncond_pred_buf);
    if (cond_enc) free(cond_enc);
    if (cond_te) free(cond_te);
    if (cond_tid) free(cond_tid);
    if (uncond_enc) free(uncond_enc);
    if (uncond_te) free(uncond_te);
    if (uncond_tid) free(uncond_tid);

    fprintf(stderr, "[server] DENOISE finished: %.1f ms (%d steps, avg %.1f ms/step)\n",
            total_denoise_ms, num_steps, (num_steps > 0) ? (total_denoise_ms / num_steps) : 0.0);
    printf("OK %.1f\n", total_denoise_ms);
    fflush(stdout);
    return 0;
}

/* ========================================================================= */
/*  Cleanup                                                                  */
/* ========================================================================= */

static void cleanup_slot(ContextSlot* slot) {
    if (!slot->active) return;

    /* 1. De-register memory handles BEFORE freeing the context */
    for (uint32_t i = 0; i < slot->numInputs; ++i) {
        if (slot->inputMemHandles[i] && g_qnn.memDeRegister) {
            g_qnn.memDeRegister(&slot->inputMemHandles[i], 1);
            slot->inputMemHandles[i] = NULL;
        }
        if (slot->inputDims[i]) {
            free(slot->inputDims[i]);
            slot->inputDims[i] = NULL;
        }
        if (slot->inputBufs[i]) {
            shared_free(slot->inputBufs[i]);
            slot->inputBufs[i] = NULL;
        }
    }
    for (uint32_t i = 0; i < slot->numOutputs; ++i) {
        if (slot->outputMemHandles[i] && g_qnn.memDeRegister) {
            g_qnn.memDeRegister(&slot->outputMemHandles[i], 1);
            slot->outputMemHandles[i] = NULL;
        }
        if (slot->outputDims[i]) {
            free(slot->outputDims[i]);
            slot->outputDims[i] = NULL;
        }
        if (slot->outputBufs[i]) {
            shared_free(slot->outputBufs[i]);
            slot->outputBufs[i] = NULL;
        }
    }

    /* 2. Free QNN context handle */
    if (slot->contextHandle && g_qnn.contextFree) {
        g_qnn.contextFree(slot->contextHandle, NULL);
        slot->contextHandle = NULL;
    }

    /* 3. Free binary data if still present */
    if (slot->binaryData) {
        free(slot->binaryData);
        slot->binaryData = NULL;
    }

    if (g_totalLoadedBytes >= slot->modelBytes) {
        g_totalLoadedBytes -= slot->modelBytes;
    } else {
        g_totalLoadedBytes = 0;
    }
    slot->modelBytes = 0;
    slot->active = 0;
}

static int cmd_unload(const char* id) {
    int si = find_slot(id);
    if (si < 0) {
        printf("ERR context_not_found %s\n", id);
        fflush(stdout);
        return -1;
    }
    cleanup_slot(&g_slots[si]);
    /* Compact the slots array */
    for (int i = si; i < g_numSlots - 1; ++i) {
        g_slots[i] = g_slots[i + 1];
    }
    g_numSlots--;
    memset(&g_slots[g_numSlots], 0, sizeof(ContextSlot));
    fprintf(stderr, "[server] Unloaded context %s\n", id);
    printf("OK\n");
    fflush(stdout);
    return 0;
}

static void cleanup_all(void) {
    for (int i = 0; i < g_numSlots; ++i) {
        cleanup_slot(&g_slots[i]);
    }

    if (g_deviceHandle && g_qnn.deviceFree) {
        g_qnn.deviceFree(g_deviceHandle);
        g_deviceHandle = NULL;
    }
    if (g_backendHandle && g_qnn.backendFree) {
        g_qnn.backendFree(g_backendHandle);
        g_backendHandle = NULL;
    }
    if (g_logHandle && g_qnn.logFree) {
        g_qnn.logFree(g_logHandle);
        g_logHandle = NULL;
    }
    /* Do NOT dlclose — avoids segfault from QNN's atexit/cleanup handlers */
}

static int ensure_fifo_path(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISFIFO(st.st_mode)) {
            return 0;
        }
        fprintf(stderr, "[server] Path exists but is not a FIFO: %s\n", path);
        return -1;
    }
    if (mkfifo(path, 0666) == 0 || errno == EEXIST) {
        return 0;
    }
    fprintf(stderr, "[server] mkfifo failed for %s: %s\n", path, strerror(errno));
    return -1;
}

static int read_command_from_fifo(const char* request_fifo, char* line, size_t line_size) {
    FILE* req = fopen(request_fifo, "r");
    if (!req) {
        fprintf(stderr, "[server] Failed to open request FIFO %s: %s\n", request_fifo, strerror(errno));
        return -1;
    }
    if (!fgets(line, (int)line_size, req)) {
        fclose(req);
        return 1;
    }
    fclose(req);

    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
        line[--len] = '\0';
    }
    return 0;
}

static int redirect_stdout_to_fifo(const char* response_fifo, int* saved_stdout_fd, FILE** response_stream) {
    fflush(stdout);
    *saved_stdout_fd = dup(STDOUT_FILENO);
    if (*saved_stdout_fd < 0) {
        fprintf(stderr, "[server] dup(stdout) failed: %s\n", strerror(errno));
        return -1;
    }

    FILE* rsp = fopen(response_fifo, "w");
    if (!rsp) {
        fprintf(stderr, "[server] Failed to open response FIFO %s: %s\n", response_fifo, strerror(errno));
        close(*saved_stdout_fd);
        *saved_stdout_fd = -1;
        return -1;
    }
    if (dup2(fileno(rsp), STDOUT_FILENO) < 0) {
        fprintf(stderr, "[server] dup2(response_fifo) failed: %s\n", strerror(errno));
        fclose(rsp);
        close(*saved_stdout_fd);
        *saved_stdout_fd = -1;
        return -1;
    }
    *response_stream = rsp;
    return 0;
}

static void restore_stdout_from_fifo(int saved_stdout_fd, FILE* response_stream) {
    fflush(stdout);
    if (saved_stdout_fd >= 0) {
        dup2(saved_stdout_fd, STDOUT_FILENO);
        close(saved_stdout_fd);
    }
    if (response_stream) {
        fclose(response_stream);
    }
}

static int dispatch_command_line(char* line) {
    char cmd[32] = {0};
    char arg1[MAX_PATH_LEN] = {0};
    char arg2[MAX_PATH_LEN] = {0};
    char arg3[MAX_PATH_LEN] = {0};

    int nargs = sscanf(line, "%31s %1023s %1023s %1023s", cmd, arg1, arg2, arg3);

    if (strcmp(cmd, "QUIT") == 0) {
        printf("OK\n");
        fflush(stdout);
        return 1;
    } else if (strcmp(cmd, "PING") == 0) {
        printf("OK %d\n", g_numSlots);
        fflush(stdout);
        return 0;
    } else if (strcmp(cmd, "LOAD") == 0 && nargs >= 3) {
        cmd_load(arg1, arg2);
    } else if (strcmp(cmd, "UNLOAD") == 0 && nargs >= 2) {
        cmd_unload(arg1);
    } else if (strcmp(cmd, "RUN") == 0 && nargs >= 4) {
        cmd_run(arg1, arg2, arg3);
    } else if (strcmp(cmd, "RUN_CHAIN") == 0) {
        /* RUN_CHAIN enc_id dec_id enc_il dec_il out_dir [enc_out:dec_in ...] */
        char lc[MAX_LINE_LEN];
        strncpy(lc, line, sizeof(lc)); lc[sizeof(lc)-1] = '\0';
        char* toks[128]; int nt = 0;
        for (char* t = strtok(lc, " \t"); t && nt < 128; t = strtok(NULL, " \t"))
            toks[nt++] = t;
        if (nt >= 6)
            cmd_run_chain(toks[1], toks[2], toks[3], toks[4], toks[5], nt - 6, &toks[6]);
        else {
            printf("ERR RUN_CHAIN needs >=5 args (got %d)\n", nt - 1);
            fflush(stdout);
        }
    } else if (strcmp(cmd, "DENOISE") == 0 && nargs >= 2) {
        cmd_denoise(arg1);
    } else {
        printf("ERR unknown_command %s\n", cmd);
        fflush(stdout);
    }
    return 0;
}

/* ========================================================================= */
/*  Standalone End-to-End CLI Generator (Zero Python / Zero Root / Zero APK) */
/* ========================================================================= */

#define VOCAB_HASH_SIZE 131072

typedef struct {
    char*   key;
    int32_t val;
} StrIntEntry;

typedef struct {
    StrIntEntry* vocab;
    StrIntEntry* merges;
    char         byte_enc[256][4];
    int32_t      bos_id;
    int32_t      eos_id;
} ClipBpeTokenizer;

static uint32_t fnv1a_str(const char* s) {
    uint32_t h = 2166136261u;
    while (*s) {
        h ^= (uint8_t)(*s++);
        h *= 16777619u;
    }
    return h;
}

static void ht_put(StrIntEntry* table, const char* key, int32_t val) {
    uint32_t idx = fnv1a_str(key) & (VOCAB_HASH_SIZE - 1);
    while (table[idx].key != NULL) {
        if (strcmp(table[idx].key, key) == 0) {
            table[idx].val = val;
            return;
        }
        idx = (idx + 1) & (VOCAB_HASH_SIZE - 1);
    }
    table[idx].key = strdup(key);
    table[idx].val = val;
}

static int32_t ht_get(const StrIntEntry* table, const char* key, int32_t def_val) {
    uint32_t idx = fnv1a_str(key) & (VOCAB_HASH_SIZE - 1);
    while (table[idx].key != NULL) {
        if (strcmp(table[idx].key, key) == 0) return table[idx].val;
        idx = (idx + 1) & (VOCAB_HASH_SIZE - 1);
    }
    return def_val;
}

static int encode_utf8_cp(uint32_t cp, char* out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        out[1] = '\0';
        return 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        out[2] = '\0';
        return 2;
    } else {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        out[3] = '\0';
        return 3;
    }
}

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

static int clip_tok_init(ClipBpeTokenizer* tok, const char* vocab_path, const char* merges_path) {
    memset(tok, 0, sizeof(*tok));
    tok->vocab  = (StrIntEntry*)calloc(VOCAB_HASH_SIZE, sizeof(StrIntEntry));
    tok->merges = (StrIntEntry*)calloc(VOCAB_HASH_SIZE, sizeof(StrIntEntry));
    tok->bos_id = 49406;
    tok->eos_id = 49407;

    /* 1. Build byte_to_unicode table */
    int in_bs[256] = {0};
    for (int b = 33; b <= 126; ++b) in_bs[b] = 1;
    for (int b = 161; b <= 172; ++b) in_bs[b] = 1;
    for (int b = 174; b <= 255; ++b) in_bs[b] = 1;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp = in_bs[b] ? (uint32_t)b : (uint32_t)(256 + extra++);
        encode_utf8_cp(cp, tok->byte_enc[b]);
    }

    /* 2. Parse vocab.json (line-by-line "key": id) */
    FILE* fv = fopen(vocab_path, "r");
    if (!fv) {
        fprintf(stderr, "[tok] ERR: cannot open %s\n", vocab_path);
        return -1;
    }
    char line[4096];
    while (fgets(line, sizeof(line), fv)) {
        char* q1 = strchr(line, '"');
        if (!q1) continue;
        char key[1024];
        int ki = 0;
        char* p = q1 + 1;
        while (*p && *p != '"' && ki < 1018) {
            if (*p == '\\') {
                p++;
                if (*p == '"' || *p == '\\' || *p == '/') key[ki++] = *p++;
                else if (*p == 'n') { key[ki++] = '\n'; p++; }
                else if (*p == 't') { key[ki++] = '\t'; p++; }
                else if (*p == 'u' && p[1] && p[2] && p[3] && p[4]) {
                    uint32_t cp = (uint32_t)((hex_val(p[1]) << 12) | (hex_val(p[2]) << 8) |
                                             (hex_val(p[3]) << 4)  |  hex_val(p[4]));
                    char u8[4];
                    int n = encode_utf8_cp(cp, u8);
                    for (int j = 0; j < n; ++j) key[ki++] = u8[j];
                    p += 5;
                } else if (*p) {
                    key[ki++] = *p++;
                }
            } else {
                key[ki++] = *p++;
            }
        }
        key[ki] = '\0';
        if (*p == '"') p++;
        char* col = strchr(p, ':');
        if (!col) continue;
        int32_t id = (int32_t)strtol(col + 1, NULL, 10);
        ht_put(tok->vocab, key, id);
    }
    fclose(fv);

    /* 3. Parse merges.txt */
    FILE* fm = fopen(merges_path, "r");
    if (!fm) {
        fprintf(stderr, "[tok] ERR: cannot open %s\n", merges_path);
        return -1;
    }
    int32_t rank = 0;
    int first_line = 1;
    while (fgets(line, sizeof(line), fm)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (first_line && strncmp(line, "#version", 8) == 0) { first_line = 0; continue; }
        first_line = 0;
        char* sp = strchr(line, ' ');
        if (!sp) continue;
        *sp = '\x01';
        ht_put(tok->merges, line, rank++);
    }
    fclose(fm);
    return 0;
}

static void clip_bpe_encode_word(const ClipBpeTokenizer* tok, const char* raw_word, int raw_len,
                                 int32_t* out_ids, int* io_count, int max_ids) {
    if (raw_len <= 0) return;

    /* Split raw bytes into UTF-8 byte_encoder symbols; append </w> to last symbol */
    char syms[256][256];
    int n_syms = 0;
    for (int i = 0; i < raw_len && n_syms < 255; ++i) {
        uint8_t b = (uint8_t)raw_word[i];
        if (i == raw_len - 1) {
            snprintf(syms[n_syms], sizeof(syms[0]), "%s</w>", tok->byte_enc[b]);
        } else {
            snprintf(syms[n_syms], sizeof(syms[0]), "%s", tok->byte_enc[b]);
        }
        n_syms++;
    }

    while (n_syms > 1) {
        int best_rank = 0x7FFFFFFF;
        int best_idx = -1;
        for (int i = 0; i < n_syms - 1; ++i) {
            char pair_key[520];
            snprintf(pair_key, sizeof(pair_key), "%s\x01%s", syms[i], syms[i + 1]);
            int32_t r = ht_get(tok->merges, pair_key, -1);
            if (r >= 0 && r < best_rank) {
                best_rank = r;
                best_idx = i;
            }
        }
        if (best_idx < 0) break;

        char first[256], second[256];
        strncpy(first, syms[best_idx], sizeof(first) - 1); first[sizeof(first) - 1] = '\0';
        strncpy(second, syms[best_idx + 1], sizeof(second) - 1); second[sizeof(second) - 1] = '\0';

        char next_syms[256][256];
        int next_n = 0;
        int i = 0;
        while (i < n_syms) {
            if (i < n_syms - 1 && strcmp(syms[i], first) == 0 && strcmp(syms[i + 1], second) == 0) {
                snprintf(next_syms[next_n++], sizeof(next_syms[0]), "%s%s", first, second);
                i += 2;
            } else {
                snprintf(next_syms[next_n++], sizeof(next_syms[0]), "%s", syms[i]);
                i += 1;
            }
        }
        n_syms = next_n;
        for (int j = 0; j < n_syms; ++j) {
            strcpy(syms[j], next_syms[j]);
        }
    }

    for (int i = 0; i < n_syms && *io_count < max_ids; ++i) {
        int32_t id = ht_get(tok->vocab, syms[i], tok->eos_id);
        out_ids[(*io_count)++] = id;
    }
}

static void clip_tok_encode(const ClipBpeTokenizer* tok, const char* text,
                            int32_t pad_id, float* out_f32_77) {
    int32_t ids[77];
    int count = 0;
    ids[count++] = tok->bos_id;

    /* Lowercase copy */
    char low[4096];
    size_t len = strlen(text);
    if (len >= sizeof(low)) len = sizeof(low) - 1;
    for (size_t i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)text[i];
        low[i] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : (char)c;
    }
    low[len] = '\0';

    const char* p = low;
    while (*p && count < 76) {
        if (isspace((unsigned char)*p)) { p++; continue; }
        /* Contractions */
        if (*p == '\'' && (p[1] == 's' || p[1] == 't' || p[1] == 'm' || p[1] == 'd') &&
            !isalpha((unsigned char)p[2])) {
            clip_bpe_encode_word(tok, p, 2, ids, &count, 76);
            p += 2;
            continue;
        }
        if (*p == '\'' && ((p[1] == 'r' && p[2] == 'e') || (p[1] == 'v' && p[2] == 'e') ||
                           (p[1] == 'l' && p[2] == 'l')) && !isalpha((unsigned char)p[3])) {
            clip_bpe_encode_word(tok, p, 3, ids, &count, 76);
            p += 3;
            continue;
        }
        /* Letters (including UTF-8 bytes >= 128) */
        if (isalpha((unsigned char)*p) || ((unsigned char)*p >= 128)) {
            const char* s = p;
            while (*p && (isalpha((unsigned char)*p) || ((unsigned char)*p >= 128))) p++;
            clip_bpe_encode_word(tok, s, (int)(p - s), ids, &count, 76);
            continue;
        }
        /* Single digit */
        if (isdigit((unsigned char)*p)) {
            clip_bpe_encode_word(tok, p, 1, ids, &count, 76);
            p++;
            continue;
        }
        /* Punctuation / non-whitespace non-alphanumeric */
        const char* s = p;
        while (*p && !isspace((unsigned char)*p) && !isalnum((unsigned char)*p) && ((unsigned char)*p < 128)) p++;
        clip_bpe_encode_word(tok, s, (int)(p - s), ids, &count, 76);
    }

    if (count < 77) ids[count++] = tok->eos_id;
    else ids[76] = tok->eos_id;

    while (count < 77) ids[count++] = pad_id;
    for (int i = 0; i < 77; ++i) out_f32_77[i] = (float)ids[i];
}

/* NumPy-exact MT19937 + polar Box-Muller randn */
typedef struct {
    uint32_t key[624];
    int      pos;
    int      has_gauss;
    double   gauss;
} NumpyRng;

static void np_rng_seed(NumpyRng* rng, uint32_t seed) {
    rng->key[0] = seed;
    for (int i = 1; i < 624; ++i) {
        rng->key[i] = (1812433253UL * (rng->key[i - 1] ^ (rng->key[i - 1] >> 30)) + (uint32_t)i);
    }
    rng->pos = 624;
    rng->has_gauss = 0;
    rng->gauss = 0.0;
}

static uint32_t np_rng_u32(NumpyRng* rng) {
    if (rng->pos == 624) {
        for (int i = 0; i < 624 - 397; ++i) {
            uint32_t y = (rng->key[i] & 0x80000000UL) | (rng->key[i + 1] & 0x7FFFFFFFUL);
            rng->key[i] = rng->key[i + 397] ^ (y >> 1) ^ ((y & 1) ? 0x9908B0DFUL : 0UL);
        }
        for (int i = 624 - 397; i < 623; ++i) {
            uint32_t y = (rng->key[i] & 0x80000000UL) | (rng->key[i + 1] & 0x7FFFFFFFUL);
            rng->key[i] = rng->key[i + (397 - 624)] ^ (y >> 1) ^ ((y & 1) ? 0x9908B0DFUL : 0UL);
        }
        uint32_t y = (rng->key[623] & 0x80000000UL) | (rng->key[0] & 0x7FFFFFFFUL);
        rng->key[623] = rng->key[396] ^ (y >> 1) ^ ((y & 1) ? 0x9908B0DFUL : 0UL);
        rng->pos = 0;
    }
    uint32_t y = rng->key[rng->pos++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680UL;
    y ^= (y << 15) & 0xEFC60000UL;
    y ^= (y >> 18);
    return y;
}

static double np_rng_double(NumpyRng* rng) {
    int32_t a = (int32_t)(np_rng_u32(rng) >> 5);
    int32_t b = (int32_t)(np_rng_u32(rng) >> 6);
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

static double np_rng_gauss(NumpyRng* rng) {
    if (rng->has_gauss) {
        rng->has_gauss = 0;
        return rng->gauss;
    }
    double x1, x2, r2;
    do {
        x1 = 2.0 * np_rng_double(rng) - 1.0;
        x2 = 2.0 * np_rng_double(rng) - 1.0;
        r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);
    double f = sqrt(-2.0 * log(r2) / r2);
    rng->gauss = f * x1;
    rng->has_gauss = 1;
    return f * x2;
}

/* Minimal PNG writer using zlib compress2 + crc32 */
static void write_be32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)((v >> 24) & 0xFF);
    p[1] = (uint8_t)((v >> 16) & 0xFF);
    p[2] = (uint8_t)((v >> 8) & 0xFF);
    p[3] = (uint8_t)(v & 0xFF);
}

static void write_png_chunk(FILE* f, const char type[4], const uint8_t* data, uint32_t len) {
    uint8_t len_be[4], crc_be[4];
    write_be32(len_be, len);
    fwrite(len_be, 1, 4, f);
    fwrite(type, 1, 4, f);
    if (len > 0 && data) fwrite(data, 1, len, f);
    uLong crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, (const Bytef*)type, 4);
    if (len > 0 && data) crc = crc32(crc, (const Bytef*)data, len);
    write_be32(crc_be, (uint32_t)crc);
    fwrite(crc_be, 1, 4, f);
}

static int save_rgb_png(const char* path, const uint8_t* rgb, int width, int height) {
    size_t row_bytes = (size_t)width * 3;
    size_t raw_size = (row_bytes + 1) * (size_t)height;
    uint8_t* raw = (uint8_t*)malloc(raw_size);
    if (!raw) return -1;
    for (int y = 0; y < height; ++y) {
        raw[y * (row_bytes + 1)] = 0; /* filter type 0 (None) */
        memcpy(raw + y * (row_bytes + 1) + 1, rgb + (size_t)y * row_bytes, row_bytes);
    }

    uLongf z_cap = compressBound((uLong)raw_size);
    uint8_t* z_buf = (uint8_t*)malloc(z_cap);
    if (!z_buf) { free(raw); return -1; }
    if (compress2(z_buf, &z_cap, raw, (uLong)raw_size, 1) != Z_OK) {
        free(z_buf);
        free(raw);
        return -1;
    }
    free(raw);

    FILE* f = fopen(path, "wb");
    if (!f) { free(z_buf); return -1; }
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    fwrite(sig, 1, 8, f);

    uint8_t ihdr[13];
    write_be32(ihdr + 0, (uint32_t)width);
    write_be32(ihdr + 4, (uint32_t)height);
    ihdr[8]  = 8; /* bit depth */
    ihdr[9]  = 2; /* color type: truecolor RGB */
    ihdr[10] = 0; /* compression */
    ihdr[11] = 0; /* filter */
    ihdr[12] = 0; /* interlace */
    write_png_chunk(f, "IHDR", ihdr, 13);
    write_png_chunk(f, "IDAT", z_buf, (uint32_t)z_cap);
    write_png_chunk(f, "IEND", NULL, 0);
    fclose(f);
    free(z_buf);
    return 0;
}

static int run_clip_pair_in_memory(ContextSlot* clip_l, ContextSlot* clip_g,
                                   const ClipBpeTokenizer* tok, const char* text,
                                   float* out_pe_77x2048, float* out_te_1280,
                                   double* out_ms_l, double* out_ms_g) {
    float ids_l[77], ids_g[77];
    clip_tok_encode(tok, text, 49407, ids_l);
    clip_tok_encode(tok, text, 0,     ids_g);

    tensor_set_f32(clip_l, 0, ids_l, 77);
    double t0 = now_ms();
    if (QNN_SUCCESS != g_qnn.graphExecute(clip_l->graphHandle, clip_l->inputs, clip_l->numInputs,
                                          clip_l->outputs, clip_l->numOutputs, NULL, NULL)) {
        return -1;
    }
    *out_ms_l = now_ms() - t0;

    tensor_set_f32(clip_g, 0, ids_g, 77);
    double t1 = now_ms();
    if (QNN_SUCCESS != g_qnn.graphExecute(clip_g->graphHandle, clip_g->inputs, clip_g->numInputs,
                                          clip_g->outputs, clip_g->numOutputs, NULL, NULL)) {
        return -1;
    }
    *out_ms_g = now_ms() - t1;

    float* cl = (float*)malloc(77 * 768 * sizeof(float));
    float* cg = (float*)malloc(77 * 1280 * sizeof(float));
    tensor_get_f32(clip_l, 0, cl, 77 * 768);
    tensor_get_f32(clip_g, 0, cg, 77 * 1280);
    tensor_get_f32(clip_g, 1, out_te_1280, 1280);

    for (int s = 0; s < 77; ++s) {
        memcpy(out_pe_77x2048 + s * 2048,       cl + s * 768,  768 * sizeof(float));
        memcpy(out_pe_77x2048 + s * 2048 + 768, cg + s * 1280, 1280 * sizeof(float));
    }
    free(cl);
    free(cg);
    return 0;
}

static inline float catmull_rom_weight(float x) {
    float ax = fabsf(x);
    if (ax <= 1.0f) {
        return 1.5f * ax * ax * ax - 2.5f * ax * ax + 1.0f;
    } else if (ax < 2.0f) {
        return -0.5f * ax * ax * ax + 2.5f * ax * ax - 4.0f * ax + 2.0f;
    }
    return 0.0f;
}

static inline int clamp_i(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

/*
 * 4x4 Catmull-Rom Bicubic Resizer + AMD FidelityFX Contrast-Adaptive Sharpening (CAS)
 * Used when target (dst_W, dst_H) exceeds the loaded NPU bucket canvas (src_W, src_H).
 */
static float* resize_rgb_catmull_rom_cas(const float* src_rgb, int src_W, int src_H,
                                         int dst_W, int dst_H, float cas_strength) {
    size_t dst_pixels = (size_t)dst_W * (size_t)dst_H;
    float* tmp_rgb = (float*)malloc(dst_pixels * 3 * sizeof(float));
    if (!tmp_rgb) return NULL;

    float scale_x = (float)src_W / (float)dst_W;
    float scale_y = (float)src_H / (float)dst_H;

    for (int dy = 0; dy < dst_H; ++dy) {
        float sy = ((float)dy + 0.5f) * scale_y - 0.5f;
        int iy = (int)floorf(sy);
        float fy = sy - (float)iy;
        float wy[4] = {
            catmull_rom_weight(fy + 1.0f),
            catmull_rom_weight(fy),
            catmull_rom_weight(1.0f - fy),
            catmull_rom_weight(2.0f - fy)
        };
        int y_idx[4] = {
            clamp_i(iy - 1, 0, src_H - 1),
            clamp_i(iy,     0, src_H - 1),
            clamp_i(iy + 1, 0, src_H - 1),
            clamp_i(iy + 2, 0, src_H - 1)
        };

        for (int dx = 0; dx < dst_W; ++dx) {
            float sx = ((float)dx + 0.5f) * scale_x - 0.5f;
            int ix = (int)floorf(sx);
            float fx = sx - (float)ix;
            float wx[4] = {
                catmull_rom_weight(fx + 1.0f),
                catmull_rom_weight(fx),
                catmull_rom_weight(1.0f - fx),
                catmull_rom_weight(2.0f - fx)
            };
            int x_idx[4] = {
                clamp_i(ix - 1, 0, src_W - 1),
                clamp_i(ix,     0, src_W - 1),
                clamp_i(ix + 1, 0, src_W - 1),
                clamp_i(ix + 2, 0, src_W - 1)
            };

            float r = 0.0f, g = 0.0f, b = 0.0f;
            for (int m = 0; m < 4; ++m) {
                const float* row = src_rgb + (size_t)y_idx[m] * (size_t)src_W * 3;
                float wy_m = wy[m];
                for (int n = 0; n < 4; ++n) {
                    float w = wy_m * wx[n];
                    const float* px = row + (size_t)x_idx[n] * 3;
                    r += w * px[0];
                    g += w * px[1];
                    b += w * px[2];
                }
            }
            float* out_px = tmp_rgb + ((size_t)dy * (size_t)dst_W + (size_t)dx) * 3;
            out_px[0] = r;
            out_px[1] = g;
            out_px[2] = b;
        }
    }

    if (cas_strength <= 0.0f) return tmp_rgb;

    /* Apply Contrast-Adaptive Sharpening (CAS) in [0, 1] normalized space */
    float* cas_rgb = (float*)malloc(dst_pixels * 3 * sizeof(float));
    if (!cas_rgb) return tmp_rgb;

    float peak = -1.0f / (8.0f - 3.0f * cas_strength);
    for (int y = 0; y < dst_H; ++y) {
        int ym1 = (y > 0) ? (y - 1) : 0;
        int yp1 = (y + 1 < dst_H) ? (y + 1) : (dst_H - 1);
        for (int x = 0; x < dst_W; ++x) {
            int xm1 = (x > 0) ? (x - 1) : 0;
            int xp1 = (x + 1 < dst_W) ? (x + 1) : (dst_W - 1);

            const float* pN = tmp_rgb + ((size_t)ym1 * dst_W + x) * 3;
            const float* pS = tmp_rgb + ((size_t)yp1 * dst_W + x) * 3;
            const float* pW = tmp_rgb + ((size_t)y * dst_W + xm1) * 3;
            const float* pE = tmp_rgb + ((size_t)y * dst_W + xp1) * 3;
            const float* pC = tmp_rgb + ((size_t)y * dst_W + x) * 3;
            float* dst_px   = cas_rgb + ((size_t)y * dst_W + x) * 3;

            for (int c = 0; c < 3; ++c) {
                /* Convert [-1, 1] -> [0, 1] for CAS min/max envelope */
                float vN = pN[c] * 0.5f + 0.5f;
                float vS = pS[c] * 0.5f + 0.5f;
                float vW = pW[c] * 0.5f + 0.5f;
                float vE = pE[c] * 0.5f + 0.5f;
                float vC = pC[c] * 0.5f + 0.5f;

                float mn = vC;
                if (vN < mn) mn = vN;
                if (vS < mn) mn = vS;
                if (vW < mn) mn = vW;
                if (vE < mn) mn = vE;
                if (mn < 0.0f) mn = 0.0f;

                float mx = vC;
                if (vN > mx) mx = vN;
                if (vS > mx) mx = vS;
                if (vW > mx) mx = vW;
                if (vE > mx) mx = vE;
                if (mx > 1.0f) mx = 1.0f;

                float d_min = mn;
                float d_max = 1.0f - mx;
                float amp = (mx > 1e-4f) ? sqrtf((d_min < d_max ? d_min : d_max) / mx) : 0.0f;
                float w = amp * peak;
                float out_v = (vC + w * (vN + vS + vW + vE)) / (1.0f + 4.0f * w);
                dst_px[c] = (out_v - 0.5f) * 2.0f;
            }
        }
    }
    free(tmp_rgb);
    return cas_rgb;
}

static int file_exists_nonempty(const char* path) {
    struct stat st;
    return (stat(path, &st) == 0 && st.st_size > 1024);
}

/*
 * Multi-Bucket Resolution Auto-Router:
 * Searches <base_dir>/context/ for the tightest compiled QNN context binary (Wg >= req_W, Hg >= req_H).
 * Falls back to the default 1024x1024 context binary if no resolution-specific binary is found.
 */
static void resolve_bucket_paths(const char* base_dir, int req_W, int req_H, const char* lora_slot,
                                 char* out_unet_path, char* out_vae_path) {
    static const int buckets[][2] = {
        {768, 768},
        {832, 1216}, {1216, 832},
        {1024, 1024},
        {1024, 1280}, {1280, 1024},
        {1152, 1536}, {1536, 1152},
        {1344, 1728}, {1728, 1344},
        {1536, 1536}
    };
    const int num_buckets = (int)(sizeof(buckets) / sizeof(buckets[0]));
    out_unet_path[0] = '\0';
    out_vae_path[0] = '\0';

    /* Optional: Check LoRA slot context */
    if (lora_slot && lora_slot[0] && strcmp(lora_slot, "None") != 0) {
        char cand_l[MAX_PATH_LEN];
        snprintf(cand_l, sizeof(cand_l), "%s/context/lora_slots/%s/unet_lightning8step_%dx%d.serialized.bin.bin", base_dir, lora_slot, req_W, req_H);
        if (file_exists_nonempty(cand_l)) {
            strcpy(out_unet_path, cand_l);
            fprintf(stderr, "[server] LoRA: selected UNet context '%s'\n", cand_l);
        } else {
            snprintf(cand_l, sizeof(cand_l), "%s/context/lora_slots/%s/unet_lightning8step.serialized.bin.bin", base_dir, lora_slot);
            if (file_exists_nonempty(cand_l)) {
                strcpy(out_unet_path, cand_l);
                fprintf(stderr, "[server] LoRA: selected UNet context '%s'\n", cand_l);
            } else {
                snprintf(cand_l, sizeof(cand_l), "%s/context/%s/unet_lightning8step.serialized.bin.bin", base_dir, lora_slot);
                if (file_exists_nonempty(cand_l)) {
                    strcpy(out_unet_path, cand_l);
                    fprintf(stderr, "[server] LoRA: selected UNet context '%s'\n", cand_l);
                } else if (file_exists_nonempty(lora_slot)) {
                    strcpy(out_unet_path, lora_slot);
                    fprintf(stderr, "[server] LoRA: selected direct context '%s'\n", lora_slot);
                } else {
                    fprintf(stderr, "[server] LoRA: slot '%s' context not found, using base model\n", lora_slot);
                }
            }
        }
    }

    /* 1. Check exact resolution match first */
    char cand_u[MAX_PATH_LEN], cand_v[MAX_PATH_LEN];
    snprintf(cand_u, sizeof(cand_u), "%s/context/unet_lightning8step_%dx%d.serialized.bin.bin", base_dir, req_W, req_H);
    snprintf(cand_v, sizeof(cand_v), "%s/context/vae_decoder_%dx%d.serialized.bin.bin", base_dir, req_W, req_H);
    if (file_exists_nonempty(cand_u) && file_exists_nonempty(cand_v)) {
        if (!out_unet_path[0]) strcpy(out_unet_path, cand_u);
        strcpy(out_vae_path, cand_v);
        return;
    }

    /* 2. Check tightest covering bucket (Wg >= req_W && Hg >= req_H) with minimal area */
    int best_area = 0x7FFFFFFF;
    char best_u[MAX_PATH_LEN] = {0}, best_v[MAX_PATH_LEN] = {0};

    for (int i = 0; i < num_buckets; ++i) {
        int bw = buckets[i][0], bh = buckets[i][1];
        if (bw < req_W || bh < req_H) continue;
        int area = bw * bh;
        if (area >= best_area) continue;

        snprintf(cand_u, sizeof(cand_u), "%s/context/unet_lightning8step_%dx%d.serialized.bin.bin", base_dir, bw, bh);
        snprintf(cand_v, sizeof(cand_v), "%s/context/vae_decoder_%dx%d.serialized.bin.bin", base_dir, bw, bh);
        if (file_exists_nonempty(cand_u) && file_exists_nonempty(cand_v)) {
            best_area = area;
            strcpy(best_u, cand_u);
            strcpy(best_v, cand_v);
            continue;
        }
        if (bw == 1024 && bh == 1024) {
            snprintf(cand_u, sizeof(cand_u), "%s/context/unet_lightning8step.serialized.bin.bin", base_dir);
            snprintf(cand_v, sizeof(cand_v), "%s/context/vae_decoder.serialized.bin.bin", base_dir);
            if (file_exists_nonempty(cand_u) && file_exists_nonempty(cand_v)) {
                best_area = area;
                strcpy(best_u, cand_u);
                strcpy(best_v, cand_v);
            }
        }
    }

    if (best_u[0] && best_v[0]) {
        if (!out_unet_path[0]) strcpy(out_unet_path, best_u);
        strcpy(out_vae_path, best_v);
        return;
    }

    /* 3. Default fallback */
    if (!out_unet_path[0]) {
        snprintf(out_unet_path, MAX_PATH_LEN, "%s/context/unet_lightning8step.serialized.bin.bin", base_dir);
    }
    snprintf(out_vae_path,  MAX_PATH_LEN, "%s/context/vae_decoder.serialized.bin.bin", base_dir);
}

static void print_npu_utilization_report(int graph_W, int graph_H, int act_W, int act_H,
                                         double total_unet_ms, double total_vae_ms) {
    int passes = (g_perf.unet_passes > 0) ? g_perf.unet_passes : 1;
    double pass_wall_ms    = total_unet_ms / passes;
    double pass_qnn_ms     = g_perf.qnn_wall_ms / passes;
    double pass_temb_ms    = g_perf.temb_proj_ms / passes;
    double pass_rpcmem_ms  = g_perf.rpcmem_bias_ms / passes;
    double pass_io_ms      = g_perf.io_quant_ms / passes;
    double pass_dev_ms     = (g_perf.qnn_device_us / 1000.0) / passes;
    double pass_dev_ex_ms  = (g_perf.qnn_device_excl_wait_us / 1000.0) / passes;
    double pass_host_rpc   = (g_perf.qnn_host_rpc_us / 1000.0) / passes;
    double pass_htp_rpc    = (g_perf.qnn_htp_rpc_us / 1000.0) / passes;
    double pass_wait_ms    = (g_perf.qnn_wait_us / 1000.0) / passes;
    uint64_t pass_cycles   = g_perf.qnn_device_cycles / (uint64_t)passes;

    /*
     * Theoretical Roofline Model for SDXL UNet on Snapdragon 8 Elite (SM8750, Hexagon V79):
     * - Base @ 1024x1024 (128x128 latent): 6,280.0 GFLOPs (6.28 TFLOPs / 3.14 TMACs) per forward pass
     *   across 60 Transformer blocks @ C=1280 (4.00 TFLOPs), 10 Transformer blocks @ C=640 (1.04 TFLOPs),
     *   and 17 ResNet Conv2d blocks (1.24 TFLOPs).
     * - Model weights read per pass: 2.44 GiB (2.618 GB) W8 + 96.99 MB U16 resnet_bias + intermediate
     *   activation DDR spill/fill across 362 layers (8 MB VTCM holds working tiles; ~13.8 GB activation traffic).
     * - Effective sustained LPDDR5X-4800MHz (9600 MT/s) NPU DMA ceiling: ~58.0 GB/s.
     * - Practical uncompressed memory bandwidth ceiling for monolithic 2.57B W8A16 UNet @ 1024x1024 is ~640 ms/pass (~9.81 TOPS).
     *   Compute units (HMX tensor core + 1024-bit HVX vector units) retain additional headroom.
     */
    double scale_area = ((double)graph_W * (double)graph_H) / (1024.0 * 1024.0);
    double gflops_pass = 6280.0 * scale_area;
    double pure_npu_ms = (pass_dev_ex_ms > 1.0) ? pass_dev_ex_ms : ((pass_dev_ms > 1.0) ? pass_dev_ms : pass_qnn_ms);
    double achieved_tops = (pure_npu_ms > 0.0) ? (gflops_pass / pure_npu_ms) : 0.0; /* GFLOPs/ms == TOPS */
    double dsp_ghz = (pure_npu_ms > 0.0 && pass_cycles > 0)
        ? ((double)pass_cycles / (pure_npu_ms * 1.0e6)) : 0.0;

    double hw_limit_ms = 640.0 * scale_area;
    double npu_roof_pct = (pure_npu_ms > 0.0) ? (100.0 * hw_limit_ms / pure_npu_ms) : 0.0;
    if (npu_roof_pct > 99.9) npu_roof_pct = 99.9;
    double pipeline_npu_duty_pct = (pass_wall_ms > 0.0) ? (100.0 * pure_npu_ms / pass_wall_ms) : 0.0;

    fprintf(stderr, "\n================ [NPU UTILIZATION & BOTTLENECK AUDIT] ================\n");
    fprintf(stderr, "Graph Canvas: %dx%d | Active Sub-Canvas: %dx%d | UNet Passes: %d",
            graph_W, graph_H, act_W, act_H, passes);
    if (g_perf.hvx_threads > 0) fprintf(stderr, " | HVX Threads: %d", g_perf.hvx_threads);
    fprintf(stderr, "\n----------------------------------------------------------------------\n");
    fprintf(stderr, "[Per-Pass Host & NPU Breakdown (avg over %d passes)]:\n", passes);
    fprintf(stderr, "  1. Host temb MLP (21 FP16 layers):  %6.2f ms (%5.1f%%) [Total: %6.1f ms]\n",
            pass_temb_ms, 100.0 * pass_temb_ms / pass_wall_ms, g_perf.temb_proj_ms);
    fprintf(stderr, "  2. Host->RPCMEM 97MB bias fill:     %6.2f ms (%5.1f%%) [Total: %6.1f ms]\n",
            pass_rpcmem_ms, 100.0 * pass_rpcmem_ms / pass_wall_ms, g_perf.rpcmem_bias_ms);
    fprintf(stderr, "  3. Host sample/enc/pred quant+xpose:%6.2f ms (%5.1f%%) [Total: %6.1f ms]\n",
            pass_io_ms, 100.0 * pass_io_ms / pass_wall_ms, g_perf.io_quant_ms);
    fprintf(stderr, "  4. QNN graphExecute (Host Wall):    %6.2f ms (%5.1f%%) [Total: %6.1f ms]\n",
            pass_qnn_ms, 100.0 * pass_qnn_ms / pass_wall_ms, g_perf.qnn_wall_ms);
    if (pass_dev_ms > 0.0 || pass_host_rpc > 0.0) {
        fprintf(stderr, "     ├─ ARM FastRPC Call Time:        %6.2f ms\n", pass_host_rpc);
        fprintf(stderr, "     ├─ HTP DSP FastRPC Time:         %6.2f ms\n", pass_htp_rpc);
        fprintf(stderr, "     ├─ VTCM / HMX+HVX Acquire Wait:  %6.2f ms\n", pass_wait_ms);
        if (pass_cycles > 0) {
            fprintf(stderr, "     └─ Pure Hexagon V79 NPU Accel:   %6.2f ms (%llu Mcycles @ %.2f GHz)\n",
                    pure_npu_ms, (unsigned long long)(pass_cycles / 1000000ULL), dsp_ghz);
        } else {
            fprintf(stderr, "     └─ Pure Hexagon V79 NPU Accel:   %6.2f ms (excl wait: %.2f ms)\n",
                    pass_dev_ms, pass_dev_ex_ms);
        }
    }
    fprintf(stderr, "  --------------------------------------------------------------------\n");
    fprintf(stderr, "  Total Per-Pass Wall Time:           %6.2f ms (100.0%%) [Total: %6.1f ms]\n",
            pass_wall_ms, total_unet_ms);
    fprintf(stderr, "----------------------------------------------------------------------\n");
    fprintf(stderr, "[Hexagon V79 LPDDR5X Memory Roofline & Utilization]:\n");
    fprintf(stderr, "  • NPU Pipeline Duty Cycle (NPU Active / Step Wall):  %5.1f%%\n", pipeline_npu_duty_pct);
    fprintf(stderr, "  • Achieved Effective Math Throughput (W8A16):        %5.2f TOPS (%.1f TFLOPs/pass)\n",
            achieved_tops, gflops_pass / 1000.0);
    fprintf(stderr, "  • Uncompressed Memory Bandwidth Roofline (vs ~%.0fms): %5.1f%%\n",
            hw_limit_ms, npu_roof_pct);
    if (g_perf.vae_device_us > 0.0) {
        fprintf(stderr, "  • VAE Decoder NPU Accel: %.1f ms device / %.1f ms wall\n",
                g_perf.vae_device_us / 1000.0, total_vae_ms);
    }
    fprintf(stderr, "======================================================================\n");
}

static int run_standalone_generate(const char* base_dir, const char* tokenizer_dir,
                                   const char* prompt, const char* neg_prompt,
                                   uint32_t seed, int steps, float cfg_scale, int cfg_cutoff_arg,
                                   int cfg_cache_mode, int req_width, int req_height, int pad_mode,
                                   const char* unet_mode, const char* lora_slot, float lora_scale,
                                   const char* out_png_path) {
    double t_total0 = now_ms();

    /* Validate & snap requested resolution to multiples of 8 within [512x512, 1536x1536] pixel budget */
    int width  = ((req_width  + 4) / 8) * 8;
    int height = ((req_height + 4) / 8) * 8;
    if (width < 256) width = 256;
    if (height < 256) height = 256;

    const int min_pixels = 512 * 512;   /* 262,144 (0.26 MP) */
    const int max_pixels = 1536 * 1536; /* 2,359,296 (2.36 MP, covers 1344x1728 = 2,322,432) */
    long long req_pixels = (long long)width * (long long)height;
    if (req_pixels < min_pixels) {
        double sc = sqrt((double)min_pixels / (double)req_pixels);
        width  = (((int)ceil(width  * sc) + 7) / 8) * 8;
        height = (((int)ceil(height * sc) + 7) / 8) * 8;
        fprintf(stderr, "[res] Clamped up to minimum SDXL pixel budget: %dx%d\n", width, height);
    } else if (req_pixels > max_pixels) {
        double sc = sqrt((double)max_pixels / (double)req_pixels);
        width  = (((int)floor(width  * sc)) / 8) * 8;
        height = (((int)floor(height * sc)) / 8) * 8;
        fprintf(stderr, "[res] Clamped down to maximum SDXL 2.36MP pixel budget: %dx%d\n", width, height);
    }

    int use_cfg = (cfg_scale > 1.0f);
    int cfg_cutoff = (use_cfg && cfg_cutoff_arg > 0 && cfg_cutoff_arg < steps) ? cfg_cutoff_arg : steps;
    if (!neg_prompt || !neg_prompt[0]) {
        neg_prompt = use_cfg ? "lowres, bad anatomy, bad hands, text, error, worst quality, low quality, blurry" : "";
    }
    const char* cache_name = (cfg_cache_mode == 1) ? "delta-cache" : ((cfg_cache_mode == 2) ? "uncond-cache" : "cfg1-after");
    const char* pad_name   = (pad_mode == 1) ? "tile" : ((pad_mode == 2) ? "zero" : "reflect");

    /* Create QNN Profile handle if --profile was requested */
    if (g_perf.enabled && g_qnn.profileCreate && !g_profHandle) {
        QnnProfile_Level_t lvl = g_perf.detailed ? QNN_PROFILE_LEVEL_DETAILED : QNN_PROFILE_LEVEL_BASIC;
        if (QNN_SUCCESS != g_qnn.profileCreate(g_backendHandle, lvl, &g_profHandle)) {
            fprintf(stderr, "[prof] WARN: QnnProfile_create failed, continuing with host timers\n");
            g_profHandle = NULL;
        }
    }

    /* Resolve tightest covering context bucket (with optional LoRA) */
    char unet_bin_path[MAX_PATH_LEN], vae_bin_path[MAX_PATH_LEN];
    resolve_bucket_paths(base_dir, width, height, lora_slot, unet_bin_path, vae_bin_path);

    fprintf(stderr, "================================================\n");
    fprintf(stderr, "[SDXL-NPU Standalone C Engine]\n");
    fprintf(stderr, "Prompt:     %s\n", prompt);
    if (lora_slot && lora_slot[0] && strcmp(lora_slot, "None") != 0) {
        fprintf(stderr, "LoRA:       %s (weight=%.2f)\n", lora_slot, lora_scale);
    }
    fprintf(stderr, "Mode:       %s (steps=%d, cfg=%.2f, cutoff=%d/%d [%s], seed=%u)\n",
            unet_mode, steps, cfg_scale, cfg_cutoff, steps, cache_name, seed);
    fprintf(stderr, "Target Res: %dx%d (%.2f MP, pad=%s)\n",
            width, height, ((double)width * height) / 1e6, pad_name);
    fprintf(stderr, "Output:     %s\n", out_png_path);
    fprintf(stderr, "================================================\n");

    /* 1. Initialize CLIP BPE Tokenizer */
    char vocab_path[MAX_PATH_LEN] = {0}, merges_path[MAX_PATH_LEN] = {0};
    int tok_ok = 0;
    if (tokenizer_dir && tokenizer_dir[0]) {
        snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", tokenizer_dir);
        snprintf(merges_path, sizeof(merges_path), "%s/merges.txt", tokenizer_dir);
        if (access(vocab_path, R_OK) == 0 && access(merges_path, R_OK) == 0) tok_ok = 1;
    }
    if (!tok_ok) {
        snprintf(vocab_path, sizeof(vocab_path), "%s/phone_gen/tokenizer/vocab.json", base_dir);
        snprintf(merges_path, sizeof(merges_path), "%s/phone_gen/tokenizer/merges.txt", base_dir);
        if (access(vocab_path, R_OK) == 0 && access(merges_path, R_OK) == 0) tok_ok = 1;
    }
    if (!tok_ok) {
        snprintf(vocab_path, sizeof(vocab_path), "%s/tokenizer/vocab.json", base_dir);
        snprintf(merges_path, sizeof(merges_path), "%s/tokenizer/merges.txt", base_dir);
        if (access(vocab_path, R_OK) == 0 && access(merges_path, R_OK) == 0) tok_ok = 1;
    }
    if (!tok_ok) {
        const char* app_tok_dirs[] = {
            "/data/user/0/com.sdxlnpu.app/files/termux_bundle/runtime_payload/tokenizer",
            "/data/user/0/com.sdxlnpu.app/files/termux_bundle/runtime_payload/phone_gen/tokenizer",
            "/data/data/com.sdxlnpu.app/files/termux_bundle/runtime_payload/tokenizer",
            "/data/data/com.sdxlnpu.app/files/termux_bundle/runtime_payload/phone_gen/tokenizer",
            NULL
        };
        for (int i = 0; !tok_ok && app_tok_dirs[i]; ++i) {
            snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", app_tok_dirs[i]);
            snprintf(merges_path, sizeof(merges_path), "%s/merges.txt", app_tok_dirs[i]);
            if (access(vocab_path, R_OK) == 0 && access(merges_path, R_OK) == 0) tok_ok = 1;
        }
    }
    if (!tok_ok) {
        snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.json", base_dir);
        snprintf(merges_path, sizeof(merges_path), "%s/merges.txt", base_dir);
        if (access(vocab_path, R_OK) == 0 && access(merges_path, R_OK) == 0) tok_ok = 1;
    }
    ClipBpeTokenizer tok;
    if (clip_tok_init(&tok, vocab_path, merges_path) != 0) {
        fprintf(stderr, "ERR: failed to load CLIP tokenizer from %s / %s\n", vocab_path, merges_path);
        return 1;
    }

    /* 2. Load & Run CLIP-L and CLIP-G */
    char clip_l_path[MAX_PATH_LEN], clip_g_path[MAX_PATH_LEN];
    snprintf(clip_l_path, sizeof(clip_l_path), "%s/context/clip_l.serialized.bin.bin", base_dir);
    snprintf(clip_g_path, sizeof(clip_g_path), "%s/context/clip_g.serialized.bin.bin", base_dir);
    if (cmd_load("clip_l", clip_l_path) != 0 || cmd_load("clip_g", clip_g_path) != 0) return 1;

    ContextSlot* s_cl = &g_slots[find_slot("clip_l")];
    ContextSlot* s_cg = &g_slots[find_slot("clip_g")];

    float* pe_cond = (float*)malloc(77 * 2048 * sizeof(float));
    float* te_cond = (float*)malloc(1280 * sizeof(float));
    float* pe_uncond = use_cfg ? (float*)malloc(77 * 2048 * sizeof(float)) : NULL;
    float* te_uncond = use_cfg ? (float*)malloc(1280 * sizeof(float)) : NULL;

    double ms_cl1 = 0, ms_cg1 = 0, ms_cl2 = 0, ms_cg2 = 0;
    if (run_clip_pair_in_memory(s_cl, s_cg, &tok, prompt, pe_cond, te_cond, &ms_cl1, &ms_cg1) != 0) return 1;
    fprintf(stderr, "[CLIP cond]   L=%.0fms G=%.0fms\n", ms_cl1, ms_cg1);
    if (use_cfg) {
        if (run_clip_pair_in_memory(s_cl, s_cg, &tok, neg_prompt, pe_uncond, te_uncond, &ms_cl2, &ms_cg2) != 0) return 1;
        fprintf(stderr, "[CLIP uncond] L=%.0fms G=%.0fms\n", ms_cl2, ms_cg2);
    }
    double total_clip_ms = ms_cl1 + ms_cg1 + ms_cl2 + ms_cg2;

    /* Unload CLIP contexts before loading UNet */
    cmd_unload("clip_g");
    cmd_unload("clip_l");

    /* 3. Load UNet (Monolithic W8A16 by default, or Split FP16 if --mode split) */
    int is_split = (strcmp(unet_mode, "split") == 0);
    if (!is_split) {
        if (cmd_load("unet", unet_bin_path) != 0) return 1;
    } else {
        char enc_path[MAX_PATH_LEN], dec_path[MAX_PATH_LEN];
        snprintf(enc_path, sizeof(enc_path), "%s/context/unet_encoder_fp16.serialized.bin.bin", base_dir);
        snprintf(dec_path, sizeof(dec_path), "%s/context/unet_decoder_fp16.serialized.bin.bin", base_dir);
        if (cmd_load("enc", enc_path) != 0 || cmd_load("dec", dec_path) != 0) return 1;
    }

    ContextSlot* enc = &g_slots[find_slot(is_split ? "enc" : "unet")];
    ContextSlot* dec = is_split ? &g_slots[find_slot("dec")] : NULL;
    ContextSlot* final_out_slot = is_split ? dec : enc;
    int is_ext_resnet = (!is_split && enc->numInputs >= 19);

    int enc_smp_idx = is_ext_resnet ? 0 : 4;
    int enc_enc_idx = is_ext_resnet ? 1 : 0;
    int enc_ts_idx  = is_ext_resnet ? -1 : 1;
    int enc_tid_idx = is_ext_resnet ? -1 : 2;
    int enc_te_idx  = is_ext_resnet ? -1 : 3;

    /* Inspect compiled UNet graph latent dimensions (graph_lat_h, graph_lat_w) */
    int graph_lat_h = 128, graph_lat_w = 128;
    get_tensor_spatial_hw(&enc->inputs[enc_smp_idx], &graph_lat_h, &graph_lat_w);
    int graph_W = graph_lat_w * 8;
    int graph_H = graph_lat_h * 8;

    /*
     * Determine active sub-canvas (act_W, act_H) centered inside (graph_W, graph_H):
     * - If width <= graph_W && height <= graph_H: 1:1 native sub-canvas (act_W = width, act_H = height).
     * - If width > graph_W || height > graph_H (and no larger bucket binary was on disk):
     *   Fit maximum aspect-ratio-preserving multiple-of-16 sub-canvas inside (graph_W, graph_H),
     *   pass true target resolution in SDXL micro-conditioning time_ids = [height, width, 0, 0, height, width],
     *   and reconstruct exact (width, height) after VAE with 4x4 Catmull-Rom + CAS!
     */
    int act_W = width, act_H = height;
    if (act_W > graph_W || act_H > graph_H) {
        double sc_w = (double)graph_W / (double)width;
        double sc_h = (double)graph_H / (double)height;
        double sc = (sc_w < sc_h) ? sc_w : sc_h;
        act_W = (((int)floor(width  * sc + 0.5)) / 16) * 16;
        act_H = (((int)floor(height * sc + 0.5)) / 16) * 16;
        if (act_W < 64) act_W = 64;
        if (act_H < 64) act_H = 64;
        if (act_W > graph_W) act_W = graph_W;
        if (act_H > graph_H) act_H = graph_H;
    }

    int act_lat_h = act_H / 8, act_lat_w = act_W / 8;
    int lat_y0 = (graph_lat_h - act_lat_h) / 2;
    int lat_x0 = (graph_lat_w - act_lat_w) / 2;
    int crop_y = lat_y0 * 8;
    int crop_x = lat_x0 * 8;

    if (width != act_W || height != act_H) {
        fprintf(stderr, "[res] Graph bucket %dx%d -> Centered Spatial-CFG Sub-Canvas %dx%d @ (%d,%d) + Catmull-Rom CAS -> %dx%d\n",
                graph_W, graph_H, act_W, act_H, crop_x, crop_y, width, height);
    } else if (act_W < graph_W || act_H < graph_H) {
        fprintf(stderr, "[res] Native 1:1 Centered Spatial-CFG Sub-Canvas: %dx%d @ (%d,%d) inside %dx%d graph\n",
                act_W, act_H, crop_x, crop_y, graph_W, graph_H);
    }

    /*
     * Natural Full-Grid Latent Evolution + Spatial CFG Framing:
     * Evolve the full [1, 4, graph_lat_h, graph_lat_w] latent coherently (zero mirror-symmetry seams,
     * 100% exact GroupNorm variance) while concentrating CFG guidance inside the centered active window
     * [lat_y0 .. lat_y0 + act_lat_h, lat_x0 .. lat_x0 + act_lat_w] and smoothly decaying to uncond (CFG=1)
     * in the inactive outer margin so the main subject is framed squarely inside [act_W, act_H].
     */
    int latent_h = graph_lat_h, latent_w = graph_lat_w;
    size_t spatial_lat = (size_t)latent_h * (size_t)latent_w;
    size_t num_latent  = (size_t)4 * spatial_lat;

    float* spatial_cfg_mask = (float*)malloc(spatial_lat * sizeof(float));
    for (int ly = 0; ly < latent_h; ++ly) {
        int dy = (ly < lat_y0) ? (lat_y0 - ly) : ((ly >= lat_y0 + act_lat_h) ? (ly - (lat_y0 + act_lat_h - 1)) : 0);
        for (int lx = 0; lx < latent_w; ++lx) {
            int dx = (lx < lat_x0) ? (lat_x0 - lx) : ((lx >= lat_x0 + act_lat_w) ? (lx - (lat_x0 + act_lat_w - 1)) : 0);
            if (dx == 0 && dy == 0) {
                spatial_cfg_mask[(size_t)ly * latent_w + lx] = 1.0f;
            } else {
                float d2 = (float)(dx * dx + dy * dy);
                spatial_cfg_mask[(size_t)ly * latent_w + lx] = expf(-0.25f * d2);
            }
        }
    }

    /* 4. Euler Discrete Scheduler + NumPy-exact Initial Latent for [1, 4, latent_h, latent_w] */
    double all_sigmas[1000];
    double alpha_cumprod = 1.0;
    double b0_sqrt = sqrt(0.00085), b1_sqrt = sqrt(0.012);
    for (int i = 0; i < 1000; ++i) {
        double bs = b0_sqrt + (b1_sqrt - b0_sqrt) * ((double)i / 999.0);
        double beta = bs * bs;
        alpha_cumprod *= (1.0 - beta);
        all_sigmas[i] = sqrt((1.0 - alpha_cumprod) / alpha_cumprod);
    }
    DenoiseScheduleStep sched[64];
    double step_ratio = 1000.0 / (double)steps;
    float sigmas[65];
    float timesteps[64];
    for (int s = 0; s < steps; ++s) {
        int t_idx = (int)llround(1000.0 - (double)s * step_ratio) - 1;
        if (t_idx < 0) t_idx = 0;
        if (t_idx > 999) t_idx = 999;
        timesteps[s] = (float)t_idx;
        sigmas[s] = (float)all_sigmas[t_idx];
    }
    sigmas[steps] = 0.0f;
    for (int s = 0; s < steps; ++s) {
        sched[s].step = s;
        sched[s].timestep = timesteps[s];
        sched[s].sigma = sigmas[s];
        sched[s].sigma_next = sigmas[s + 1];
    }
    float init_noise_sigma = sigmas[0];

    NumpyRng rng;
    np_rng_seed(&rng, seed);
    float* latent = (float*)malloc(num_latent * sizeof(float));
    for (size_t i = 0; i < num_latent; ++i) {
        latent[i] = (float)np_rng_gauss(&rng) * init_noise_sigma;
    }

    /* SDXL Micro-Conditioning: [orig_H, orig_W, crop_top=0, crop_left=0, target_H, target_W] */
    float tid[6] = {(float)height, (float)width, 0.0f, 0.0f, (float)height, (float)width};

    typedef struct { int enc_out_idx; int dec_in_idx; size_t copy_size; } PipeMap;
    PipeMap pipes[16];
    int num_pipes = 0;
    if (is_split && dec) {
        static const char* pipe_pairs[11][2] = {
            {"output_0", "mid_out"}, {"output_1", "skip_0"}, {"output_2", "skip_1"},
            {"output_3", "skip_2"},  {"output_4", "skip_3"}, {"output_5", "skip_4"},
            {"output_6", "skip_5"},  {"output_7", "skip_6"}, {"output_8", "skip_7"},
            {"output_9", "skip_8"},  {"output_10", "temb"}
        };
        for (int p = 0; p < 11; ++p) {
            int eidx = -1, didx = -1;
            for (uint32_t j = 0; j < enc->numOutputs; ++j)
                if (strcmp(enc->outputNames[j], pipe_pairs[p][0]) == 0) { eidx = (int)j; break; }
            for (uint32_t j = 0; j < dec->numInputs; ++j)
                if (strcmp(dec->inputNames[j], pipe_pairs[p][1]) == 0) { didx = (int)j; break; }
            if (eidx >= 0 && didx >= 0) {
                pipes[num_pipes].enc_out_idx = eidx;
                pipes[num_pipes].dec_in_idx = didx;
                pipes[num_pipes].copy_size = enc->outputBufSizes[eidx];
                num_pipes++;
            }
        }
    }

    float* scaled_sample   = (float*)malloc(num_latent * sizeof(float));
    float* cond_pred_buf   = (float*)malloc(num_latent * sizeof(float));
    float* uncond_pred_buf = use_cfg ? (float*)malloc(num_latent * sizeof(float)) : NULL;

    double t_unet0 = now_ms();
    for (int s = 0; s < steps; ++s) {
        double ts0 = now_ms();
        DenoiseScheduleStep st = sched[s];
        int step_cfg = (use_cfg && s < cfg_cutoff);
        float inv = 1.0f / sqrtf(st.sigma * st.sigma + 1.0f);
        for (size_t i = 0; i < num_latent; ++i) scaled_sample[i] = latent[i] * inv;

        unet_set_sample_nchw(enc, (uint32_t)enc_smp_idx, scaled_sample, latent_h, latent_w);
        if (enc_ts_idx >= 0) tensor_set_f32(enc, (uint32_t)enc_ts_idx, &st.timestep, 1);

        if (step_cfg) {
            /* Uncond */
            if (is_ext_resnet) {
                if (compute_and_set_unet_resnet_biases(enc, st.timestep, te_uncond, tid) != 0) return 1;
            } else {
                tensor_set_f32(enc, (uint32_t)enc_tid_idx, tid, 6);
                tensor_set_f32(enc, (uint32_t)enc_te_idx, te_uncond, 1280);
            }
            unet_set_enc_hidden(enc, (uint32_t)enc_enc_idx, pe_uncond);
            if (is_split && dec) unet_set_enc_hidden(dec, 0, pe_uncond);
            double tq0 = now_ms();
            g_qnn.graphExecute(enc->graphHandle, enc->inputs, enc->numInputs, enc->outputs, enc->numOutputs, g_profHandle, NULL);
            g_perf.qnn_wall_ms += (now_ms() - tq0);
            g_perf.unet_passes++;
            if (g_profHandle) collect_qnn_profile_events(g_profHandle, 0);
            if (is_split && dec) {
                for (int p = 0; p < num_pipes; ++p)
                    memcpy(dec->inputBufs[pipes[p].dec_in_idx], enc->outputBufs[pipes[p].enc_out_idx], pipes[p].copy_size);
                g_qnn.graphExecute(dec->graphHandle, dec->inputs, dec->numInputs, dec->outputs, dec->numOutputs, g_profHandle, NULL);
            }
            unet_get_noise_pred_nchw(final_out_slot, 0, uncond_pred_buf, latent_h, latent_w);

            /* Cond */
            if (is_ext_resnet) {
                compute_and_set_unet_resnet_biases(enc, st.timestep, te_cond, tid);
            } else {
                tensor_set_f32(enc, (uint32_t)enc_tid_idx, tid, 6);
                tensor_set_f32(enc, (uint32_t)enc_te_idx, te_cond, 1280);
            }
            unet_set_enc_hidden(enc, (uint32_t)enc_enc_idx, pe_cond);
            if (is_split && dec) unet_set_enc_hidden(dec, 0, pe_cond);
            double tq1 = now_ms();
            g_qnn.graphExecute(enc->graphHandle, enc->inputs, enc->numInputs, enc->outputs, enc->numOutputs, g_profHandle, NULL);
            g_perf.qnn_wall_ms += (now_ms() - tq1);
            g_perf.unet_passes++;
            if (g_profHandle) collect_qnn_profile_events(g_profHandle, 0);
            if (is_split && dec) {
                for (int p = 0; p < num_pipes; ++p)
                    memcpy(dec->inputBufs[pipes[p].dec_in_idx], enc->outputBufs[pipes[p].enc_out_idx], pipes[p].copy_size);
                g_qnn.graphExecute(dec->graphHandle, dec->inputs, dec->numInputs, dec->outputs, dec->numOutputs, g_profHandle, NULL);
            }
            unet_get_noise_pred_nchw(final_out_slot, 0, cond_pred_buf, latent_h, latent_w);

            float delta = st.sigma_next - st.sigma;
            float base_boost = cfg_scale - 1.0f;
            for (int c = 0; c < 4; ++c) {
                size_t ch_off = (size_t)c * spatial_lat;
                for (size_t p = 0; p < spatial_lat; ++p) {
                    size_t i = ch_off + p;
                    float d_cfg = cond_pred_buf[i] - uncond_pred_buf[i];
                    float w_px = 1.0f + base_boost * spatial_cfg_mask[p];
                    float guided = uncond_pred_buf[i] + w_px * d_cfg;
                    latent[i] += delta * guided;
                    if (cfg_cache_mode == 1 && s == cfg_cutoff - 1) {
                        uncond_pred_buf[i] = d_cfg * spatial_cfg_mask[p];
                    }
                }
            }
        } else {
            if (is_ext_resnet) {
                if (compute_and_set_unet_resnet_biases(enc, st.timestep, te_cond, tid) != 0) return 1;
            } else {
                tensor_set_f32(enc, (uint32_t)enc_tid_idx, tid, 6);
                tensor_set_f32(enc, (uint32_t)enc_te_idx, te_cond, 1280);
            }
            unet_set_enc_hidden(enc, (uint32_t)enc_enc_idx, pe_cond);
            if (is_split && dec) unet_set_enc_hidden(dec, 0, pe_cond);
            double tq2 = now_ms();
            g_qnn.graphExecute(enc->graphHandle, enc->inputs, enc->numInputs, enc->outputs, enc->numOutputs, g_profHandle, NULL);
            g_perf.qnn_wall_ms += (now_ms() - tq2);
            g_perf.unet_passes++;
            if (g_profHandle) collect_qnn_profile_events(g_profHandle, 0);
            if (is_split && dec) {
                for (int p = 0; p < num_pipes; ++p)
                    memcpy(dec->inputBufs[pipes[p].dec_in_idx], enc->outputBufs[pipes[p].enc_out_idx], pipes[p].copy_size);
                g_qnn.graphExecute(dec->graphHandle, dec->inputs, dec->numInputs, dec->outputs, dec->numOutputs, g_profHandle, NULL);
            }
            unet_get_noise_pred_nchw(final_out_slot, 0, cond_pred_buf, latent_h, latent_w);

            float delta = st.sigma_next - st.sigma;
            if (use_cfg && cfg_cache_mode == 1 && uncond_pred_buf && cfg_cutoff > 0) {
                /* Sigma-Damped Delta-Cache: guided = cond + (w - 1) * decay * cached_delta */
                float sigma_ref = sched[cfg_cutoff - 1].sigma;
                float decay = (sigma_ref > 1e-5f) ? (0.35f * (st.sigma / sigma_ref)) : 0.0f;
                float w_eff = (cfg_scale - 1.0f) * decay;
                for (size_t i = 0; i < num_latent; ++i) {
                    float guided = cond_pred_buf[i] + w_eff * uncond_pred_buf[i];
                    latent[i] += delta * guided;
                }
            } else if (use_cfg && cfg_cache_mode == 2 && uncond_pred_buf) {
                /* Uncond-Cache: guided = cached_uncond + w * (cond - cached_uncond) */
                float base_boost = cfg_scale - 1.0f;
                for (int c = 0; c < 4; ++c) {
                    size_t ch_off = (size_t)c * spatial_lat;
                    for (size_t p = 0; p < spatial_lat; ++p) {
                        size_t i = ch_off + p;
                        float w_px = 1.0f + base_boost * spatial_cfg_mask[p];
                        float guided = uncond_pred_buf[i] + w_px * (cond_pred_buf[i] - uncond_pred_buf[i]);
                        latent[i] += delta * guided;
                    }
                }
            } else {
                for (size_t i = 0; i < num_latent; ++i) latent[i] += delta * cond_pred_buf[i];
            }
        }
        const char* step_tag = step_cfg ? " CFG" : ((use_cfg && cfg_cache_mode == 1) ? " CFG-dCache" : ((use_cfg && cfg_cache_mode == 2) ? " CFG-uCache" : ""));
        fprintf(stderr, "  [UNet %d/%d]%s %.0fms\n", s + 1, steps, step_tag, now_ms() - ts0);
    }
    double total_unet_ms = now_ms() - t_unet0;
    free(spatial_cfg_mask);

    if (is_split) {
        cmd_unload("dec");
        cmd_unload("enc");
    }

    /* 5. Load & Run VAE Decoder */
    if (cmd_load("vae", vae_bin_path) != 0) return 1;
    ContextSlot* vae = &g_slots[find_slot("vae")];

    const float inv_scaling = 1.0f / 0.13025f;
    for (size_t i = 0; i < num_latent; ++i) scaled_sample[i] = latent[i] * inv_scaling;
    unet_set_sample_nchw(vae, 0, scaled_sample, latent_h, latent_w);

    double t_vae0 = now_ms();
    if (QNN_SUCCESS != g_qnn.graphExecute(vae->graphHandle, vae->inputs, vae->numInputs,
                                          vae->outputs, vae->numOutputs, g_profHandle, NULL)) {
        return 1;
    }
    double total_vae_ms = now_ms() - t_vae0;
    g_perf.vae_wall_ms = total_vae_ms;
    if (g_profHandle) collect_qnn_profile_events(g_profHandle, 1);
    fprintf(stderr, "[VAE] %.0fms\n", total_vae_ms);

    /* Extract centered active [act_H, act_W, 3] RGB sub-image from VAE output */
    size_t act_rgb_elems = (size_t)act_W * (size_t)act_H * 3;
    float* img_f32 = (float*)malloc(act_rgb_elems * sizeof(float));
    vae_get_rgb_subrect(vae, 0, img_f32, crop_y, crop_x, act_H, act_W);

    /* If target (width, height) > bucket (act_W, act_H), reconstruct via 4x4 Catmull-Rom + CAS */
    if (width != act_W || height != act_H) {
        double tr0 = now_ms();
        float* hi_f32 = resize_rgb_catmull_rom_cas(img_f32, act_W, act_H, width, height, 0.65f);
        if (hi_f32) {
            free(img_f32);
            img_f32 = hi_f32;
            fprintf(stderr, "[CatmullRom+CAS] %dx%d -> %dx%d in %.1fms\n",
                    act_W, act_H, width, height, now_ms() - tr0);
        }
    }

    size_t num_rgb = (size_t)width * (size_t)height * 3;

    /* Normalize [-1, 1] -> [0, 1] and compute [0.5%, 99.5%] contrast stretch via 4096-bin histogram */
    uint32_t hist[4096] = {0};
    for (size_t i = 0; i < num_rgb; ++i) {
        float v = img_f32[i] * 0.5f + 0.5f;
        if (v < 0.0f) v = 0.0f;
        else if (v > 1.0f) v = 1.0f;
        img_f32[i] = v;
        int bin = (int)(v * 4095.0f + 0.5f);
        if (bin < 0) bin = 0; else if (bin > 4095) bin = 4095;
        hist[bin]++;
    }
    uint32_t target_lo = (uint32_t)(num_rgb * 0.005);
    uint32_t target_hi = (uint32_t)(num_rgb * 0.995);
    uint32_t acc = 0;
    int bin_lo = 0, bin_hi = 4095;
    for (int b = 0; b < 4096; ++b) {
        acc += hist[b];
        if (acc >= target_lo) { bin_lo = b; break; }
    }
    acc = 0;
    for (int b = 0; b < 4096; ++b) {
        acc += hist[b];
        if (acc >= target_hi) { bin_hi = b; break; }
    }
    float lo = (float)bin_lo / 4095.0f;
    float hi = (float)bin_hi / 4095.0f;
    float range = (hi - lo > 0.05f) ? (hi - lo) : 1.0f;
    float base_lo = (hi - lo > 0.05f) ? lo : 0.0f;

    uint8_t* rgb_u8 = (uint8_t*)malloc(num_rgb);
    for (size_t i = 0; i < num_rgb; ++i) {
        float v = (img_f32[i] - base_lo) / range;
        if (v < 0.0f) v = 0.0f;
        else if (v > 1.0f) v = 1.0f;
        rgb_u8[i] = (uint8_t)(v * 255.0f + 0.5f);
    }

    if (save_rgb_png(out_png_path, rgb_u8, width, height) != 0) {
        fprintf(stderr, "ERR: failed to write PNG %s\n", out_png_path);
        return 1;
    }

    if (g_perf.enabled) {
        print_npu_utilization_report(graph_W, graph_H, act_W, act_H, total_unet_ms, total_vae_ms);
    }

    double total_wall_s = (now_ms() - t_total0) / 1000.0;
    fprintf(stderr, "\n========================================\n");
    fprintf(stderr, "Saved: %s (%dx%d)\n", out_png_path, width, height);
    fprintf(stderr, "CLIP: %.0fms | UNet (%s): %.0fms (%.0fms/step) | VAE: %.0fms\n",
            total_clip_ms, unet_mode, total_unet_ms, total_unet_ms / steps, total_vae_ms);
    fprintf(stderr, "Total Wall Time: %.2fs\n", total_wall_s);
    fprintf(stderr, "Total: %.2fs\n", total_wall_s);
    fprintf(stderr, "========================================\n");

    if (g_profHandle && g_qnn.profileFree) {
        g_qnn.profileFree(g_profHandle);
        g_profHandle = NULL;
    }

    free(rgb_u8);
    free(img_f32);
    free(scaled_sample);
    free(cond_pred_buf);
    if (uncond_pred_buf) free(uncond_pred_buf);
    free(latent);
    free(pe_cond);
    free(te_cond);
    if (pe_uncond) free(pe_uncond);
    if (te_uncond) free(te_uncond);
    return 0;
}

/* ========================================================================= */
/*  Main                                                                     */
/* ========================================================================= */

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --backend <libQnnHtp.so> --system_lib <libQnnSystem.so> [options]\n"
        "\nPersistent multi-context QNN server & Standalone SDXL CLI Engine.\n"
        "\nStandalone CLI Generation Options:\n"
        "  --prompt <text>         Generate SDXL image directly in C (no Python/root/APK)\n"
        "  --neg <text>            Negative prompt (optional)\n"
        "  --width <int>           Target width (multiple of 8, default: 1024)\n"
        "  --height <int>          Target height (multiple of 8, default: 1024)\n"
        "  --res <WxH>             Target resolution shorthand (e.g. 832x1216, 768x1024, 1344x1728)\n"
        "  --pad_mode <mode>       Sub-canvas boundary isolation: reflect (default), tile, zero\n"
        "  --seed <uint>           Random seed (default: 42)\n"
        "  --steps <int>           Denoising steps (default: 8)\n"
        "  --cfg <float>           CFG guidance scale (default: 3.5, 1.0 = no CFG)\n"
        "  --prog_cfg              Progressive CFG (5/8 steps + sigma-damped delta-cache)\n"
        "  --cfg_cutoff <int>      Number of initial steps with full 2-pass CFG (1..steps)\n"
        "  --cfg_cache <mode>      After cutoff: none (CFG=1), delta (reuse cond-uncond), uncond (reuse uncond)\n"
        "  --full_cfg              Full CFG on all steps (default)\n"
        "  --profile               Print NPU hardware utilization & host bottleneck audit\n"
        "  --legacy_temb           Disable NEON + 64KB block-doubling RPCMEM optimization (for A/B test)\n"
        "  --mode <mono|split>     UNet mode: mono (W8A16 monolithic, default) or split\n"
        "  --lora <name|path>      LoRA slot name or direct context binary path\n"
        "  --lora_scale <float>    LoRA strength scale (default: 1.0)\n"
        "  --rpc_lib <path>        Path to libcdsprpc.so (for FastRPC/rpcmem initialization)\n"
        "  --base_dir <path>       SDXL base dir (default: /sdcard/Download/sdxl_qnn)\n"
        "  --out <png_path>        Output PNG file path\n"
        "\nServer Options:\n"
        "  --request_fifo <path>   Optional request FIFO for shared-server mode\n"
        "  --response_fifo <path>  Optional response FIFO for shared-server mode\n", prog);
}

int main(int argc, char** argv) {
    const char* backend_path = "libQnnHtp.so";
    const char* system_path = "libQnnSystem.so";
    const char* request_fifo = NULL;
    const char* response_fifo = NULL;
    const char* prompt = NULL;
    const char* neg_prompt = NULL;
    const char* tokenizer_dir = NULL;
    const char* unet_mode = "mono";
    const char* lora_slot = NULL;
    float lora_scale = 1.0f;
    const char* base_dir = "/sdcard/Download/sdxl_qnn";
    const char* out_png = "/sdcard/Download/sdxl_qnn/outputs/standalone_out.png";
    uint32_t seed = 42;
    int steps = 8;
    float cfg_scale = 3.5f;
    int cfg_cutoff = 0; /* 0 = full CFG (all steps) */
    int cfg_cache_mode = 0; /* 0 = none (CFG=1 after cutoff), 1 = delta, 2 = uncond */
    int req_width = 1024;
    int req_height = 1024;
    int pad_mode = 0; /* 0 = reflect, 1 = tile, 2 = zero */

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend_path = argv[++i];
        } else if (strcmp(argv[i], "--system_lib") == 0 && i + 1 < argc) {
            system_path = argv[++i];
        } else if (strcmp(argv[i], "--request_fifo") == 0 && i + 1 < argc) {
            request_fifo = argv[++i];
        } else if (strcmp(argv[i], "--response_fifo") == 0 && i + 1 < argc) {
            response_fifo = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--neg") == 0 && i + 1 < argc) {
            neg_prompt = argv[++i];
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            req_width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            req_height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--res") == 0 && i + 1 < argc) {
            const char* rstr = argv[++i];
            int rw = 0, rh = 0;
            if (sscanf(rstr, "%dx%d", &rw, &rh) == 2 || sscanf(rstr, "%dX%d", &rw, &rh) == 2) {
                req_width = rw;
                req_height = rh;
            }
        } else if (strcmp(argv[i], "--pad_mode") == 0 && i + 1 < argc) {
            const char* pm = argv[++i];
            if (strcmp(pm, "tile") == 0) pad_mode = 1;
            else if (strcmp(pm, "zero") == 0) pad_mode = 2;
            else pad_mode = 0;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cfg") == 0 && i + 1 < argc) {
            cfg_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--prog_cfg") == 0 || strcmp(argv[i], "--prog-cfg") == 0) {
            cfg_cutoff = -1; /* resolve to 5/8 steps + sigma-damped delta-cache after parsing */
            if (cfg_cache_mode == 0) cfg_cache_mode = 1;
        } else if (strcmp(argv[i], "--cfg_cutoff") == 0 && i + 1 < argc) {
            cfg_cutoff = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cfg_cache") == 0 && i + 1 < argc) {
            const char* cm = argv[++i];
            if (strcmp(cm, "delta") == 0) cfg_cache_mode = 1;
            else if (strcmp(cm, "uncond") == 0) cfg_cache_mode = 2;
            else cfg_cache_mode = 0;
        } else if (strcmp(argv[i], "--full_cfg") == 0 || strcmp(argv[i], "--no_prog_cfg") == 0) {
            cfg_cutoff = 0;
            cfg_cache_mode = 0;
        } else if (strcmp(argv[i], "--profile") == 0) {
            g_perf.enabled = 1;
        } else if (strcmp(argv[i], "--profile_detailed") == 0) {
            g_perf.enabled = 1;
            g_perf.detailed = 1;
        } else if (strcmp(argv[i], "--legacy_temb") == 0) {
            g_use_legacy_temb = 1;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            unet_mode = argv[++i];
        } else if ((strcmp(argv[i], "--lora") == 0 || strcmp(argv[i], "--lora_slot") == 0) && i + 1 < argc) {
            lora_slot = argv[++i];
        } else if (strcmp(argv[i], "--lora_scale") == 0 && i + 1 < argc) {
            lora_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--rpc_lib") == 0 && i + 1 < argc) {
            g_rpc_lib_path = argv[++i];
        } else if (strcmp(argv[i], "--tokenizer_dir") == 0 && i + 1 < argc) {
            tokenizer_dir = argv[++i];
        } else if (strcmp(argv[i], "--base_dir") == 0 && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_png = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
    }
    if (cfg_cutoff == -1) cfg_cutoff = (steps * 5 + 4) / 8;

    if (!backend_path || !system_path) {
        usage(argv[0]);
        return 1;
    }

    if ((request_fifo && !response_fifo) || (!request_fifo && response_fifo)) {
        fprintf(stderr, "[server] --request_fifo and --response_fifo must be provided together\n");
        return 1;
    }

    if (request_fifo && ensure_fifo_path(request_fifo) != 0) {
        return 1;
    }
    if (response_fifo && ensure_fifo_path(response_fifo) != 0) {
        return 1;
    }

    /* Init QNN */
    fprintf(stderr, "[server] Initializing QNN...\n");
    if (init_qnn(backend_path, system_path, base_dir) != 0) {
        fprintf(stderr, "[server] QNN initialization failed\n");
        return 1;
    }

    /* Set HTP performance mode */
    set_perf_mode();

    /* Standalone CLI Generation Mode */
    if (prompt != NULL) {
        int rc = run_standalone_generate(base_dir, tokenizer_dir, prompt, neg_prompt, seed, steps, cfg_scale,
                                         cfg_cutoff, cfg_cache_mode, req_width, req_height,
                                         pad_mode, unet_mode, lora_slot, lora_scale, out_png);
        cleanup_all();
        return rc;
    }

    fprintf(stderr, "[server] Ready (backend=%s)\n", backend_path);
    printf("READY\n");
    fflush(stdout);

    /* Command loop */
    char line[MAX_LINE_LEN];
    if (request_fifo && response_fifo) {
        while (1) {
            int rr = read_command_from_fifo(request_fifo, line, sizeof(line));
            if (rr != 0) {
                continue;
            }

            if (line[0] == '\0') {
                continue;
            }

            int saved_stdout_fd = -1;
            FILE* response_stream = NULL;
            if (redirect_stdout_to_fifo(response_fifo, &saved_stdout_fd, &response_stream) != 0) {
                continue;
            }
            int should_quit = dispatch_command_line(line);
            restore_stdout_from_fifo(saved_stdout_fd, response_stream);
            if (should_quit) {
                break;
            }
        }
    } else {
        while (fgets(line, sizeof(line), stdin)) {
            /* Strip trailing newline */
            size_t len = strlen(line);
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
                line[--len] = '\0';
            }
            if (len == 0) continue;
            if (dispatch_command_line(line)) {
                break;
            }
        }
    }

    cleanup_all();
    fprintf(stderr, "[server] Shutdown complete\n");
    return 0;
}
