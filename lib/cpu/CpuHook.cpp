#include "cpu/CpuHook.h"
#include "hook/CFuncHook.h"
#include "utils/BackTrace.h"
#include "utils/Utils.h"
#include "utils/Log/Log.h"
#include "hook/LocalHook/LocalHook.h"
#include <chrono>
#include "utils/ProfilingAccumulator/ProfilingAccumulator.h"
#include "utils/Timer/Timer.h"
#include "utils/Recorder/Recorder.h"
#include <stdarg.h>

using namespace cpu_hook;
using namespace std::chrono;

int64_t get_rank() {
    const char *rank_str= std::getenv("RANK");
    if (rank_str == nullptr) {
        return 0;
    }
    int64_t rank = std::stoi(rank_str);
    return rank;
}

static int (*origin_xpu_wait_)(void *) = nullptr;
class CpuHookWrapper {
  public:
    CpuHookWrapper() {}
    ~CpuHookWrapper();
    static int local_launch_async(void *func);
    static int local_launch_config(int nclusters, int ncores, void *stream);
    static int local_launch_arg_set(const void *arg, size_t size,
                                    size_t offset);
    static int local_xpu_wait(void *stream);
    static void *local_dlsym(void *handle, const char *symbol);
    static void *local_dlopen(const char *filename, int flag);
    static int local_print(const char *format, ...);
    static int local_fprintf(void* stream, const char *format, ...);
    static int local_accumulated_fprintf(void* stream, const char *format, ...);
    static int local_event_record(void *event, void *stream);
    static int local_stream_wait_event(void *stream, void *event);
    static int local_cudaStreamWaitEvent(void*, void*, unsigned int);
    static int local_cudaStreamRecord(void*, void*);
    static int local_cudaMalloc(void** devPtr, size_t size);


    int (*origin_launch_async_)(void *) = nullptr;
    int (*origin_launch_config_)(int, int, void *) = nullptr;
    int (*origin_launch_arg_set_)(const void *, size_t, size_t) = nullptr;
    // int (*origin_xpu_wait_)(void *) = nullptr;
    void *(*origin_dlsym_)(void *, const char *) = nullptr;
    void *(*origin_dlopen_)(const char *, int) = nullptr;
    int (*origin_print_)(const char*, ...)  = nullptr;
    int (*origin_fprintf_)(void* , const char*, ...)  = nullptr;
    int (*origin_event_record_)(void *, void *) = nullptr;
    int (*origin_stream_wait_event_)(void *, void*) = nullptr;
    int (*origin_cudaStreamWaitEvent_)(void*, void*, unsigned int) = nullptr;
    int (*origin_cudaStreamRecord_)(void*, void*) = nullptr;

    int (*origin_cudaMalloc_)(void**, size_t) = nullptr;

    static std::string runtime_api_name;
};

std::string CpuHookWrapper::runtime_api_name = "";

typedef utils::Singleton<CpuHookWrapper> SingletonCpuHookWrapper;

CpuHookWrapper::~CpuHookWrapper() {}

int CpuHookWrapper::local_launch_async(void *func) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    if (wrapper_instance->origin_launch_async_ != nullptr) {
        timer::record_time(/*ph=*/"B", /*name=*/__FUNCTION__, /*runtime api=*/CpuHookWrapper::runtime_api_name);
        int s = wrapper_instance->origin_launch_async_(func);
        timer::record_time(/*ph=*/"E", /*name=*/__FUNCTION__, /*runtime api=*/CpuHookWrapper::runtime_api_name);
        return s;
    }
    return 0;
}

int CpuHookWrapper::local_launch_config(int nclusters, int ncores,
                                        void *stream) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    uintptr_t stream_index = reinterpret_cast<uintptr_t>(stream);
    std::string runtime_api = "stream: ";
    CpuHookWrapper::runtime_api_name = runtime_api + std::to_string(stream_index);
    if (wrapper_instance->origin_launch_config_ != nullptr) {
        timer::record_time(/*ph=*/"B", /*name=*/__FUNCTION__, /*runtime api=*/CpuHookWrapper::runtime_api_name);
        int ret = wrapper_instance->origin_launch_config_(nclusters, ncores, stream);
        timer::record_time(/*ph=*/"E", /*name=*/__FUNCTION__, /*runtime api=*/CpuHookWrapper::runtime_api_name);
        return ret;
    }
    return 0;
}

int CpuHookWrapper::local_launch_arg_set(const void *arg, size_t size,
                                         size_t offset) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    if (wrapper_instance->origin_launch_arg_set_ != nullptr) {
        return wrapper_instance->origin_launch_arg_set_(arg, size, offset);
    }
    return 0;
}

int CpuHookWrapper::local_event_record(void *event, void *stream) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    uintptr_t stream_index = reinterpret_cast<uintptr_t>(stream);
    std::string runtime_api = "stream: ";
    std::string runtime_api_line = runtime_api + std::to_string(stream_index);
    if (wrapper_instance->origin_event_record_ != nullptr) {
        timer::record_time(/*ph=*/"B", /*name=*/__FUNCTION__, /*runtime api=*/runtime_api_line);
        int ret = wrapper_instance->origin_event_record_(event, stream);
        auto time = timer::record_time(/*ph=*/"E", /*name=*/__FUNCTION__, /*runtime api=*/runtime_api_line);
        timer::record_flow_event(/*time=*/time, /*ph=*/"s", /*name=*/__FUNCTION__, /*runtime api=*/runtime_api_line);
        return ret;
    }
    return 0;
}

int CpuHookWrapper::local_stream_wait_event(void *stream, void *event) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    uintptr_t stream_index = reinterpret_cast<uintptr_t>(stream);
    std::string runtime_api = "stream: ";
    std::string runtime_api_line = runtime_api + std::to_string(stream_index);
    if (wrapper_instance->origin_stream_wait_event_ != nullptr) {
        auto time = timer::record_time(/*ph=*/"B", /*name=*/__FUNCTION__, /*runtime api=*/runtime_api_line);
        int ret = wrapper_instance->origin_stream_wait_event_(stream, event);
        timer::record_time(/*ph=*/"E", /*name=*/__FUNCTION__, /*runtime api=*/runtime_api_line);
        timer::record_flow_event(/*time=*/time, /*ph=*/"f", /*name=*/__FUNCTION__, /*runtime api=*/runtime_api_line);
        return ret;
    }
    return 0;
}

int CpuHookWrapper::local_cudaStreamRecord(void* event, void* stream) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    uintptr_t stream_index = reinterpret_cast<uintptr_t>(stream);
    std::string runtime_api = "stream: ";
    std::string runtime_api_line = runtime_api + std::to_string(stream_index);
    if (wrapper_instance->origin_cudaStreamRecord_ != nullptr) {
        // LOG() << "cudaStreamRecord";
        timer::record_time(/*ph=*/"B", /*name=*/__FUNCTION__, /*runtime api=*/runtime_api_line);
        int ret = wrapper_instance->origin_cudaStreamRecord_(event, stream);
        auto time = timer::record_time(/*ph=*/"E", /*name=*/__FUNCTION__, /*runtime api=*/runtime_api_line);
        timer::record_flow_event(/*time=*/time, /*ph=*/"s", /*name=*/"connect", /*runtime api=*/runtime_api_line);
        return ret;
    }

    return 0;
}

int CpuHookWrapper::local_cudaStreamWaitEvent(void *stream, void *event, unsigned int flags) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    uintptr_t stream_index = reinterpret_cast<uintptr_t>(stream);
    std::string runtime_api = "stream: ";
    std::string runtime_api_line = runtime_api + std::to_string(stream_index);
    if (wrapper_instance->origin_cudaStreamWaitEvent_ != nullptr) {
        // LOG() << "cudaStreamWaitEvent";
        auto time = timer::record_time(/*ph=*/"B", /*name=*/__FUNCTION__, /*runtime api=*/runtime_api_line);
        int ret = wrapper_instance->origin_cudaStreamWaitEvent_(stream, event, flags);
        timer::record_time(/*ph=*/"E", /*name=*/__FUNCTION__, /*runtime api=*/runtime_api_line);
        timer::record_flow_event_end(/*time=*/time, /*ph=*/"f", /*name=*/"connect", /*runtime api=*/runtime_api_line);
        return ret;
    }
    return 0;
}
int CpuHookWrapper::local_xpu_wait(void *stream) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    if (origin_xpu_wait_ != nullptr) {
        timer::record_time(/*ph=*/"B", /*name=*/__FUNCTION__, /*runtime api=*/CpuHookWrapper::runtime_api_name, /*cname=*/"good");
        int status = origin_xpu_wait_(stream);
        timer::record_time(/*ph=*/"E", /*name=*/__FUNCTION__, /*runtime api=*/CpuHookWrapper::runtime_api_name, /*cname=*/"good");
        return status;
    } else {
        ELOG() << "origin xpu wait is null";
    }
    return 0;
}

void* CpuHookWrapper::local_dlsym(void *handle, const char *symbol) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    if (wrapper_instance->origin_dlsym_ != nullptr) {
        void* ptr = wrapper_instance->origin_dlsym_(handle, symbol);
        return ptr;
    } else {
        ELOG() << "origin local dlsym is nullptr";
    }
    return nullptr;
}

void* CpuHookWrapper::local_dlopen(const char *filename, int flag) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    DLOG() << "dlopen file_name: " << filename;
    if (wrapper_instance->origin_dlopen_ != nullptr) {
        void* ptr = wrapper_instance->origin_dlopen_(filename, flag);
        return ptr;
    } else {
        ELOG() << "origin local dlsym is nullptr";
    }
}

int (*Target_launch_async)(void *func) = nullptr;
int local_launch_async(void *func) {
    DLOG() << "execute local_launch_async";
    return Target_launch_async(func);
}

int CpuHookWrapper::local_print(const char* format, ...) {
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    if (wrapper_instance->origin_print_ != nullptr) {
        va_list args;
        va_start(args, format);
        int ret = wrapper_instance->origin_print_(format, args);
        va_end(args);
        return ret;
    } else {
        ELOG() << "origin local print is nullptr";
    }
    return 0;
}


int CpuHookWrapper::local_fprintf(void* stream, const char* format, ...) {
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    std::string format_str(format);
    if (format_str.find("[XPURT_PROF]") != std::string::npos) {
        va_list args;
        va_start(args, format);
        const char* func_name = va_arg(args, const char*);
        int cycles = va_arg(args, int);
        int time = va_arg(args, int);
        timer::record_time_pair(time, func_name, "kernel", "good");
        std::string perf_str = "[XPURT_PROF] ";
        perf_str += func_name;
        perf_str += ": ";
        perf_str += std::to_string(time) + " ns";
        recorder::record(perf_str);
    }  else {
        if (wrapper_instance->origin_fprintf_ != nullptr) {
            va_list args;
            va_start(args, format);
            wrapper_instance->origin_fprintf_(stream, format, args);
            va_end(args);
        } else {
            ELOG() << "origin local fprintf is nullptr";
        }
    }
    return 0;
}


int CpuHookWrapper::local_accumulated_fprintf(void* stream, const char* format, ...) {
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    std::string format_str(format);
    if (format_str.find("[XPURT_PROF]") != std::string::npos) {
        va_list args;
        va_start(args, format);
        const char* func_name = va_arg(args, const char*);
        int cycles = va_arg(args, int);
        int duration_ns = va_arg(args, int);
        profiling_accumulator::accumulate_profiling_info(func_name, duration_ns, cycles);
    } else if (wrapper_instance->origin_fprintf_ != nullptr) {
        va_list args;
        va_start(args, format);
        int ret = wrapper_instance->origin_fprintf_(stream, format, args);
        va_end(args);
        return ret;
    } else {
        ELOG() << "origin local fprintf is nullptr";
    }
    return 0;
}
int CpuHookWrapper::local_cudaMalloc(void** devPtr, size_t size) {
    trace::Tracer tracer(__FUNCTION__);
    std::cout << "size: " << size << std::endl;
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    int ret = 0;
    if (wrapper_instance->origin_cudaMalloc_ != nullptr) {
        ret = wrapper_instance->origin_cudaMalloc_(devPtr, size);
    } else {
        ELOG() << "origin local cudaMalloc is nullptr";
    }
    return ret;
}

REGISTERHOOK(cfunc_hook::HookType::kPROFILE, kPROFILE, xpu_launch_async, (void *)CpuHookWrapper::local_launch_async,
             (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_launch_async_);
REGISTERHOOK(cfunc_hook::HookType::kPROFILE, kPROFILE, xpu_launch_config, (void *)CpuHookWrapper::local_launch_config,
             (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_launch_config_);
// REGISTERHOOK(cfunc_hook::HookType::kPROFILE, xpu_launch_argument_set,
//              (void *)CpuHookWrapper::local_launch_arg_set,
//              (void **)&SingletonCpuHookWrapper::instance()
//                  .get_elem()
//                  ->origin_launch_arg_set_);

REGISTERHOOK(cfunc_hook::HookType::kPROFILE, kPROFILE, xpu_event_record, (void *)CpuHookWrapper::local_event_record,
             (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_event_record_);

REGISTERHOOK(cfunc_hook::HookType::kPROFILE, kPROFILE, xpu_stream_event_wait,
             (void *)CpuHookWrapper::local_stream_wait_event,
             (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_stream_wait_event_);

REGISTERHOOK(cfunc_hook::HookType::kPROFILE, kPROFILE, cudaEventRecord, (void *)CpuHookWrapper::local_cudaStreamRecord,
             (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_cudaStreamRecord_);

REGISTERHOOK(cfunc_hook::HookType::kPROFILE, kPROFILE, cudaStreamWaitEvent,
             (void *)CpuHookWrapper::local_cudaStreamWaitEvent,
             (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_cudaStreamWaitEvent_);

REGISTERHOOK(cfunc_hook::HookType::kPROFILE, kPROFILE, xpu_wait, (void *)CpuHookWrapper::local_xpu_wait,
             (void **)&origin_xpu_wait_);

// REGISTERHOOK(cfunc_hook::HookType::kDEBUG, kDEBUG, cudaMalloc, (void *)CpuHookWrapper::local_cudaMalloc,
//              (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_cudaMalloc_);


// REGISTERHOOK(cfunc_hook::HookType::kPROFILE,
//     dlsym, (void *)CpuHookWrapper::local_dlsym,
//     (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_dlsym_);

// REGISTERHOOK(cfunc_hook::HookType::kPROFILE,
//     dlopen, (void *)CpuHookWrapper::local_dlopen,
//     (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_dlopen_);

// REGISTERHOOK(cfunc_hook::HookType::kPROFILE, printf, (void *)CpuHookWrapper::local_print,
//     (void**)&SingletonCpuHookWrapper::instance().get_elem()->origin_print_);

// REGISTER_LOCAL_HOOK(printf,  (void *)CpuHookWrapper::local_print,
//     (void**)&SingletonCpuHookWrapper::instance().get_elem()->origin_print_);

REGISTERHOOK(cfunc_hook::HookType::kPROFILE, kPROFILE, fprintf, (void *)CpuHookWrapper::local_fprintf,
             (void**)&SingletonCpuHookWrapper::instance().get_elem()->origin_fprintf_);

REGISTERHOOK(cfunc_hook::HookType::kACCUMULATE_KERNEL_TIME, kACCUMULATE_KERNEL_TIME, fprintf,
             (void *)CpuHookWrapper::local_accumulated_fprintf,
             (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_fprintf_);

namespace cpu_hook {
void register_cpu_hook() {
    // this function do nothing, but can not remove
}
} // namespace cpu_hook
