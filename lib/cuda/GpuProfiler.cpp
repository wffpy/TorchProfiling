#include "cuda/GpuProfiler.h"
#include <cuda.h>
#include <cupti.h>
#include "CFuncHook.h"
#include "Utils.h"

using namespace gpu_profiler;
// typedef utils::Singleton<GpuHookWrapper> SingletonGpuHookWrapper;

typedef struct cupti_eventData_st {
  CUpti_EventGroup eventGroup;
  CUpti_EventID eventId;
} cupti_eventData;

// Structure to hold data collected by callback
typedef struct RuntimeApiTrace_st {
  cupti_eventData *eventData;
  uint64_t eventVal;
} RuntimeApiTrace_t;

static void displayEventVal(RuntimeApiTrace_t *trace, const char *eventName) {
  printf("Event Name : %s \n", eventName);
  printf("Event Value : %llu\n", (unsigned long long) trace->eventVal);
}

int GpuHookWrapper::local_cuda_launch(void* func) {
//    CUcontext context = 0;
//    CUdevice dev = 0;
//    CUresult err;
//    int deviceNum = 0;
//    int deviceCount;
//    char deviceName[256];
//    const char *eventName;
//    uint32_t profile_all = 1;
//
//    CUptiResult cuptiErr;
//    CUpti_SubscriberHandle subscriber;
//    cupti_eventData cuptiEvent;
//    RuntimeApiTrace_t trace;
//
//    // check device number
//    err = cuDeviceGetCount(&deviceCount);
//    // CHECK_CU_ERROR(err, "cuDeviceGetCount");
//
//    if (deviceCount == 0) {
//        printf("There is no device supporting CUDA.\n");
//        exit(0);
//    }
//
//    std::cout << "device number: " << deviceNum << std::endl;
//
//    err = cuDeviceGet(&dev, deviceNum);
//    // CHECK_CU_ERROR(err, "cuDeviceGet");
//
//    err = cuDeviceGetName(deviceName, 256, dev);
//    // CHECK_CU_ERROR(err, "cuDeviceGetName");
//
//    std::cout << "device name: " << deviceName << std::endl;
//
//    // err = cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
//
//
//    err = cuCtxCreate(&context, 0, dev);
//
//    cuptiErr = cuptiEventGroupCreate(context, &cuptiEvent.eventGroup, 0);
//    // CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupCreate");
//
//    eventName = "cuda launch";
//
//    cuptiErr = cuptiEventGetIdFromName(dev, eventName, &cuptiEvent.eventId);
//    if (cuptiErr != CUPTI_SUCCESS) {
//        printf("Invalid eventName: %s\n", eventName);
//        exit(EXIT_FAILURE);
//    }
//
//    cuptiErr = cuptiEventGroupAddEvent(cuptiEvent.eventGroup, cuptiEvent.eventId);
//    // CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupAddEvent");
//
//    cuptiErr = cuptiEventGroupSetAttribute(cuptiEvent.eventGroup,
//                                            CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
//                                            sizeof(profile_all), &profile_all);
//    // CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupSetAttribute");
//
//    trace.eventData = &cuptiEvent;
//   
//    // TODO
//    // cuptiErr = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getEventValueCallback , &trace);
//    // CHECK_CUPTI_ERROR(cuptiErr, "cuptiSubscribe");
//
//    cuptiErr = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
//                                    CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020);
//    // CHECK_CUPTI_ERROR(cuptiErr, "cuptiEnableCallback");
//    cuptiErr = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
//                                    CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
//    // CHECK_CUPTI_ERROR(cuptiErr, "cuptiEnableCallback");
//
//    // call cudaLaunch
//    GpuHookWrapper* wrapper_instance = SingletonGpuHookWrapper::instance().get_elem();
//    if (wrapper_instance->oriign_cuda_launch_) {
//	std::cout << "cuda launch !!!!!!!!!!" << std::endl;
//        wrapper_instance->oriign_cuda_launch_(func);
//    } else {
//	std::cout << "not cuda launch !!!!!!!!!!" << std::endl;
//    }
//
//    displayEventVal(&trace, eventName);
//
//    trace.eventData = NULL;
//
//    cuptiErr = cuptiEventGroupRemoveEvent(cuptiEvent.eventGroup, cuptiEvent.eventId);
//    // CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupRemoveEvent");
//
//    cuptiErr = cuptiEventGroupDestroy(cuptiEvent.eventGroup);
//    // CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDestroy");
//
//    cuptiErr = cuptiUnsubscribe(subscriber);
//    // CHECK_CUPTI_ERROR(cuptiErr, "cuptiUnsubscribe");
//
//    cudaDeviceSynchronize();

    return 0;
}
// int GpuHookWrapper::local_cuda_launch_kernel(void* func, void* gridDim, void* blockDim, void** args, size_t sharedMem, void* stream) {
int GpuHookWrapper::local_cuda_launch_kernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    // call cudaLaunch
    GpuHookWrapper* wrapper_instance = SingletonGpuHookWrapper::instance().get_elem();
    if (wrapper_instance->oriign_cuda_launch_kernel_) {
        std::cout << "cuda launch !!!!!!!!!!" << std::endl;
        wrapper_instance->oriign_cuda_launch_kernel_(func, gridDim, blockDim, args, sharedMem, stream);
    } else {
        std::cout << "not cuda launch !!!!!!!!!!" << std::endl;
    }

    return 0;
}


// REGISTERHOOK(cudaLaunchKernel, (void *)GpuHookWrapper::local_cuda_launch_kernel,
//             (void **)&SingletonGpuHookWrapper::instance().get_elem()->oriign_cuda_launch_kernel_);
