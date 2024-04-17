#include "cuda/GpuProfiler.h"
#include "hook/LocalHook/LocalHook.h"
#include "hook/CFuncHook.h"
#include <link.h>

#ifdef CUDA_DEV
#include <cuda.h>
#include <cupti.h>
#include <cupti_events.h>
#endif

#include "hook/CFuncHook.h"
#include "utils/Utils.h"
#include "utils/BackTrace.h"
#include "utils/Log/Log.h"

using namespace gpu_profiler;

#ifdef CUDA_DEV
class GpuHookWrapper {
  public:
    GpuHookWrapper() {}
    ~GpuHookWrapper() {}

    static int local_cuda_launch_kernel_exc(const void* config, const void *func, void **args);
    static int local_cuda_launch(void *func);
    static int local_cuda_launch_kernel(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem, cudaStream_t stream);
    static int local_cuda_launch_device(const void *func, void* param_buffer, dim3 gridDim,
                                        dim3 blockDim,
                                        unsigned int sharedMemSize, cudaStream_t stream);
    static int local_cuda_graph_launch(cudaGraphExec_t graphExec, cudaStream_t stream);
    // static int cuGetProcAddress_v2(const char *symbol, void **funcPtr, int driverVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);

    int (*oriign_cuda_launch_)(void *) = nullptr;
    int (*oriign_cuda_launch_kernel_)(const void *, dim3, dim3, void **, size_t,
                                      cudaStream_t) = nullptr;
    int (*origin_cuda_launch_kernel_exc_)(const void* config, const void *func, void **args);
    int (*origin_cuda_launch_device_)(const void *, void*, dim3,
                                        dim3,
                                        unsigned int, cudaStream_t);
    int (*origin_cuda_graph_launch_)(cudaGraphExec_t, cudaStream_t);
};

typedef utils::Singleton<GpuHookWrapper> SingletonGpuHookWrapper;
#endif

#define CUPTI_CALL(call)                                                       \
    do {                                                                       \
        CUptiResult _status = call;                                            \
        if (_status != CUPTI_SUCCESS) {                                        \
            const char *errstr;                                                \
            cuptiGetResultString(_status, &errstr);                            \
            fprintf(stderr,                                                    \
                    "%s:%d: error: function %s failed with error %s.\n",       \
                    __FILE__, __LINE__, #call, errstr);                        \
            exit(-1);                                                          \
        }                                                                      \
    } while (0)

/*****************************/

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
    (((uintptr_t)(buffer) & ((align)-1))                                       \
         ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))          \
         : (buffer))

#ifdef CUDA_DEV
static void print_activity(CUpti_Activity *record) {
    switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
        CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;
        LOG() << "[XPURT_PROF] " << kernel->name << ": "
                  << kernel->end - kernel->start << " ns";
        break;
    }
    default:
        ELOG() << "unkonwn activity";
        break;
    }
}

void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size,
                               size_t *maxNumRecords) {
    uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
    if (bfr == NULL) {
        ELOG() <<  "Error: out of memory";
    }

    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
}

void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t streamId,
                               uint8_t *buffer, size_t size, size_t validSize) {
    CUptiResult status;
    CUpti_Activity *record = NULL;

    if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                print_activity(record);
            } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            else {
                CUPTI_CALL(status);
            }
        } while (1);

        // report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0) {
            LOG() << "dropped " << (unsigned int)dropped << " activity records";
        }
    }

    free(buffer);
}

void init_trace() {
    // A kernel executing on the GPU. The corresponding activity record
    // structure is CUpti_ActivityKernel4.
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

    // Register callbacks for buffer requests and for buffers completed by
    // CUPTI.
    CUPTI_CALL(
        cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
}

void fini_trace() {
    // Force flush any remaining activity buffers before termination of the
    // application
    CUPTI_CALL(cuptiActivityFlushAll(1));
    cuptiFinalize();
}

int GpuHookWrapper::local_cuda_launch_kernel(const void *func, dim3 gridDim,
                                             dim3 blockDim, void **args,
                                             size_t sharedMem,
                                             cudaStream_t stream) {
    trace::Tracer tracer(__FUNCTION__);
    cudaDeviceSynchronize();
    init_trace();
    // call cudaLaunchKernel
    GpuHookWrapper *wrapper_instance =
        SingletonGpuHookWrapper::instance().get_elem();
    if (wrapper_instance->oriign_cuda_launch_kernel_) {
        wrapper_instance->oriign_cuda_launch_kernel_(func, (gridDim), blockDim,
                                                     args, sharedMem, stream);
    } else {
        ELOG() << "Error: cudaLaunchKernel is not found";
    }
    cudaDeviceSynchronize();
    fini_trace();

    return 0;
}
int GpuHookWrapper::local_cuda_launch_device(const void *func, void* param_buffer, dim3 gridDim,
                                        dim3 blockDim,
                                        unsigned int sharedMemSize, cudaStream_t stream) {
    GpuHookWrapper *wrapper_instance =
        SingletonGpuHookWrapper::instance().get_elem();
    if (wrapper_instance->oriign_cuda_launch_kernel_) {
        wrapper_instance->origin_cuda_launch_device_(func, param_buffer,gridDim, blockDim, sharedMemSize, stream);
    } else {
        ELOG() << "Error: cudaLaunchDevice is not found";
    }
    return 0;
}
int GpuHookWrapper::local_cuda_graph_launch(cudaGraphExec_t graphExec, cudaStream_t stream) {
    GpuHookWrapper *wrapper_instance =
        SingletonGpuHookWrapper::instance().get_elem();
    if (wrapper_instance->origin_cuda_graph_launch_) {
        wrapper_instance->origin_cuda_graph_launch_(graphExec, stream);
    } else {
        ELOG() << "Error: cudaGraphLaunch is not found";
    }
    return 0;

}

int GpuHookWrapper::local_cuda_launch(void *func) {
    trace::Tracer tracer(__FUNCTION__);
    GpuHookWrapper *wrapper_instance =
        SingletonGpuHookWrapper::instance().get_elem();
    if (wrapper_instance->oriign_cuda_launch_) {
        wrapper_instance->oriign_cuda_launch_(func);
    } else {
        ELOG() << "Error: cudaLaunch is not found";
    }
    return 0;
}

int GpuHookWrapper::local_cuda_launch_kernel_exc(const void* config, const void *func, void **args) {
    trace::Tracer tracer(__FUNCTION__);
    GpuHookWrapper *wrapper_instance =
        SingletonGpuHookWrapper::instance().get_elem();
    if (wrapper_instance->origin_cuda_launch_kernel_exc_) {
        wrapper_instance->origin_cuda_launch_kernel_exc_(func, func, args);
    } else {
        ELOG() << "Error: cudaLaunchKernelExC is not found";
    }
    return 0;
}


// REGISTERHOOK(cudaLaunchKernel, (void *)GpuHookWrapper::local_cuda_launch_kernel,
//              (void **)&SingletonGpuHookWrapper::instance()
//                  .get_elem()
//                  ->oriign_cuda_launch_kernel_);

// REGISTERHOOK(cudaLaunchDevice , (void *)GpuHookWrapper::local_cuda_launch_device,
//              (void **)&SingletonGpuHookWrapper::instance()
//                  .get_elem()
//                  ->origin_cuda_launch_device_);
// REGISTERHOOK(cudaGraphLaunch, (void *)GpuHookWrapper::local_cuda_graph_launch,
//              (void **)&SingletonGpuHookWrapper::instance()
//                  .get_elem()
//                  ->origin_cuda_graph_launch_);

// REGISTERHOOK(cudaLaunch, (void *)GpuHookWrapper::local_cuda_launch,
//              (void **)&SingletonGpuHookWrapper::instance()
//                  .get_elem()
//                  ->oriign_cuda_launch_);

// REGISTERHOOK(cudaLaunchKernelExC, (void *)GpuHookWrapper::local_cuda_launch_kernel_exc,
//              (void **)&SingletonGpuHookWrapper::instance()
//                  .get_elem()
//                  ->origin_cuda_launch_kernel_exc_);
int(*Target_cudaLaunchKernel)(const void *, dim3, dim3, void **, size_t, cudaStream_t) = nullptr;

int local_cuda_launch_kernel2(const void *func, dim3 gridDim,
                                             dim3 blockDim, void **args,
                                             size_t sharedMem,
                                             cudaStream_t stream) {
    Target_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    return 0;
}

int (*Target_cuGetProcAddress)(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags, void* symbolStatus ) = nullptr;
int local_cuGetProcAddress(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags, void* symbolStatus ){
    DLOG() << __FUNCTION__ << " get symbol: " << symbol;
    return Target_cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
}

// // CUresult (*Target_cuLaunchKernel)(CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY,
// CUresult (*Target_cuLaunchKernel)(void* f, unsigned int  gridDimX, unsigned int  gridDimY,
//                                   unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY,
//                                   unsigned int  blockDimZ, unsigned int  sharedMemBytes, void* hStream,
//                                 //   void* kernelParams, void* extra) = nullptr;
//                                   void*** kernelParams, void*** extra) = nullptr;
// // CUresult local_cuLaunchKernel(CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, 
// CUresult local_cuLaunchKernel(void* f, unsigned int  gridDimX, unsigned int  gridDimY, 
//                             unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY,
//                             unsigned int  blockDimZ, unsigned int  sharedMemBytes, void* hStream,
//                             // void* kernelParams, void* extra ) {
//                             void*** kernelParams, void*** extra ) {
//     // std::cout << "enter local_cuLaunchKernel!!!!!!!!!!!!!!!!!1" << std::endl;
//     CUresult ret = Target_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
//     cudaDeviceSynchronize();
//     // std::cout << "exit local_cuLaunchKernel" << std::endl;
//     return ret;
//     // return CUDA_SUCCESS;
//     // return Target_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
// }

CUresult (*Target_cuLaunchKernel)(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra) = nullptr;
CUresult local_cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra) {
    cudaDeviceSynchronize();
    init_trace();
    std::cout << "call cuLaunch" << std::endl;

    CUresult ret = Target_cuLaunchKernel(f, 
                                gridDimX,
                                gridDimY,
                                gridDimZ,
                                blockDimX,
                                blockDimY,
                                blockDimZ,
                                sharedMemBytes,
                                hStream,
                                kernelParams,
                                extra);
    cudaDeviceSynchronize();
    fini_trace();
    return ret;
}

// REGISTER_LOCAL_HOOK(cuGetProcAddress_v2, (void*)local_cuGetProcAddress, (void**)&Target_cuGetProcAddress);
REGISTER_LOCAL_HOOK(cuLaunchKernel, (void*)local_cuLaunchKernel, (void**)&Target_cuLaunchKernel);

#endif

namespace gpu_profiler {
void register_gpu_hook() {
    // this function do nothing, but can not remove
// #ifdef CUDA_DEV
//     // REGISTER_LOCAL_HOOK(cuGetProcAddress_v2, (void*)local_cuGetProcAddress, (void**)&Target_cuGetProcAddress);
//     REGISTER_LOCAL_HOOK(cuLaunchKernel, (void*)local_cuLaunchKernel, (void**)&Target_cuLaunchKernel);
// #endif

}
} // namespace gpu_profiler
