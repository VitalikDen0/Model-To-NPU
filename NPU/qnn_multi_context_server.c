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
#include "QnnProperty.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

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

static void* g_rpcmem_lib = NULL;
static rpcmem_alloc_fn_t  g_rpcmem_alloc  = NULL;
static rpcmem_free_fn_t   g_rpcmem_free   = NULL;
static rpcmem_to_fd_fn_t  g_rpcmem_to_fd  = NULL;
static rpcmem_init_fn_t   g_rpcmem_init   = NULL;
static rpcmem_deinit_fn_t g_rpcmem_deinit = NULL;
static int g_rpcmem_available = 0;

static void init_rpcmem(void) {
    g_rpcmem_lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!g_rpcmem_lib) {
        fprintf(stderr, "[server] rpcmem: libcdsprpc.so not found, trying librpcmem.so\n");
        g_rpcmem_lib = dlopen("librpcmem.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!g_rpcmem_lib) {
        fprintf(stderr, "[server] rpcmem: not available, using regular malloc\n");
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

static int read_file_to_buf(const char* path, void* buf, size_t expected_size) {
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

static int init_qnn(const char* backend_path, const char* system_path) {
    init_rpcmem();
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
    dcvs.dcvsV3Config.dcvsEnable = 1;
    dcvs.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    dcvs.dcvsV3Config.setSleepDisable = 1;
    dcvs.dcvsV3Config.sleepDisable = 1;
    dcvs.dcvsV3Config.setBusParams = 1;
    dcvs.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
    dcvs.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    dcvs.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    dcvs.dcvsV3Config.setCoreParams = 1;
    dcvs.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
    dcvs.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
    dcvs.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    const QnnHtpPerfInfrastructure_PowerConfig_t* configs[] = { &dcvs, NULL };
    err = htpInfra->perfInfra.setPowerConfig(g_powerConfigId, configs);
    if (QNN_SUCCESS == err) {
        fprintf(stderr, "[server] HTP performance mode set (powerConfigId=%u)\n", g_powerConfigId);
    } else {
        fprintf(stderr, "[server] WARN: setPowerConfig failed: %d\n", (int)err);
    }
}

/* ========================================================================= */
/*  Context loading                                                          */
/* ========================================================================= */

/* Extract graph info from context binary via system API */
static int get_graph_info_from_binary(const void* data, size_t size,
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

    /* read the context binary into malloc'd buffer */
    if (load_file_malloc(context_path, &slot->binaryData, &slot->binarySize) != 0) {
        printf("ERR cannot_open %s: %s\n", context_path, strerror(errno));
        fflush(stdout);
        return -1;
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
                uint16_t* p = (uint16_t*)dst;
                for (size_t e = 0; e < rd; e++) p[e] = (uint16_t)floatBuf[e];
            } else if (dt == QNN_DATATYPE_UFIXED_POINT_8) {  /* 0x0408 */
                uint8_t* p = (uint8_t*)dst;
                for (size_t e = 0; e < rd; e++) p[e] = (uint8_t)floatBuf[e];
            } else if (dt == QNN_DATATYPE_SFIXED_POINT_16) { /* 0x0316 */
                int16_t* p = (int16_t*)dst;
                for (size_t e = 0; e < rd; e++) p[e] = (int16_t)floatBuf[e];
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
/*  Cleanup                                                                  */
/* ========================================================================= */

static void cleanup_slot(ContextSlot* slot) {
    if (!slot->active) return;

    if (slot->contextHandle && g_qnn.contextFree) {
        g_qnn.contextFree(slot->contextHandle, NULL);
    }

    for (uint32_t i = 0; i < slot->numInputs; ++i) {
        if (slot->inputMemHandles[i] && g_qnn.memDeRegister)
            g_qnn.memDeRegister(&slot->inputMemHandles[i], 1);
        free(slot->inputDims[i]);
        shared_free(slot->inputBufs[i]);
    }
    for (uint32_t i = 0; i < slot->numOutputs; ++i) {
        if (slot->outputMemHandles[i] && g_qnn.memDeRegister)
            g_qnn.memDeRegister(&slot->outputMemHandles[i], 1);
        free(slot->outputDims[i]);
        shared_free(slot->outputBufs[i]);
    }

    if (slot->binaryData) {
        free(slot->binaryData);
        slot->binaryData = NULL;
    }

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
    } else {
        printf("ERR unknown_command %s\n", cmd);
        fflush(stdout);
    }
    return 0;
}

/* ========================================================================= */
/*  Main                                                                     */
/* ========================================================================= */

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --backend <libQnnHtp.so> --system_lib <libQnnSystem.so>\n"
        "\nPersistent multi-context QNN server.\n"
        "Reads commands from stdin/stdout by default, or via optional shared FIFOs.\n"
        "\nCommands:\n"
        "  PING\n"
        "  LOAD <id> <context_binary_path>\n"
        "  UNLOAD <id>\n"
        "  RUN <id> <input_list_path> <output_dir>\n"
        "  RUN_CHAIN <enc_id> <dec_id> <enc_input_list> <dec_input_list> <output_dir> [enc_out:dec_in ...]\n"
        "  QUIT\n"
        "\nOptions:\n"
        "  --request_fifo <path>   Optional request FIFO for shared-server mode\n"
        "  --response_fifo <path>  Optional response FIFO for shared-server mode\n", prog);
}

int main(int argc, char** argv) {
    const char* backend_path = NULL;
    const char* system_path = NULL;
    const char* request_fifo = NULL;
    const char* response_fifo = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend_path = argv[++i];
        } else if (strcmp(argv[i], "--system_lib") == 0 && i + 1 < argc) {
            system_path = argv[++i];
        } else if (strcmp(argv[i], "--request_fifo") == 0 && i + 1 < argc) {
            request_fifo = argv[++i];
        } else if (strcmp(argv[i], "--response_fifo") == 0 && i + 1 < argc) {
            response_fifo = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        }
    }

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
    if (init_qnn(backend_path, system_path) != 0) {
        fprintf(stderr, "[server] QNN initialization failed\n");
        return 1;
    }

    /* Set HTP performance mode */
    set_perf_mode();

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
