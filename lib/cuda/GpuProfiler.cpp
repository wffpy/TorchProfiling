#include "cuda/GpuProfiler.h"

#ifdef CUDA_DEV
#include <cuda.h>
#include <cupti.h>
#include <cupti_events.h>
#endif

#include "hook/CFuncHook.h"
#include "utils/Utils.h"

using namespace gpu_profiler;

#ifdef CUDA_DEV
class GpuHookWrapper {
  public:
    GpuHookWrapper() {}
    ~GpuHookWrapper() {}

    static int local_cuda_launch(void *func);
    static int local_cuda_launch_kernel(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem, cudaStream_t stream);

    int (*oriign_cuda_launch_)(void *) = nullptr;
    int (*oriign_cuda_launch_kernel_)(const void *, dim3, dim3, void **, size_t,
                                      cudaStream_t) = nullptr;
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
        std::cout << "[CUPTI_PROF] " << kernel->name << ": "
                  << kernel->end - kernel->start << " ns" << std::endl;
        break;
    }
    default:
        std::cout << "<unknown activity>" << std::endl;
        exit(-1);
        break;
    }
}

void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size,
                               size_t *maxNumRecords) {
    uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
    if (bfr == NULL) {
        std::cout << "Error: out of memory" << std::endl;
        exit(-1);
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
            std::cout << "Dropped " << (unsigned int)dropped
                      << " activity records" << std::endl;
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
    cudaDeviceSynchronize();
    init_trace();
    // call cudaLaunchKernel
    GpuHookWrapper *wrapper_instance =
        SingletonGpuHookWrapper::instance().get_elem();
    if (wrapper_instance->oriign_cuda_launch_kernel_) {
        wrapper_instance->oriign_cuda_launch_kernel_(func, (gridDim), blockDim,
                                                     args, sharedMem, stream);
    } else {
        std::cout << "not cuda launch !!!!!!!!!!" << std::endl;
    }
    cudaDeviceSynchronize();
    fini_trace();

    return 0;
}

REGISTERHOOK(cudaLaunchKernel, (void *)GpuHookWrapper::local_cuda_launch_kernel,
             (void **)&SingletonGpuHookWrapper::instance()
                 .get_elem()
                 ->oriign_cuda_launch_kernel_);
#endif

namespace gpu_profiler {
void register_gpu_hook() {
    // this function do nothing, but can not remove
}
} // namespace gpu_profiler