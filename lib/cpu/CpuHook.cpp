#include "cpu/CpuHook.h"
#include "hook/CFuncHook.h"
#include "utils/BackTrace.h"
#include "utils/Utils.h"
#include "utils/Log/Log.h"
#include "hook/LocalHook/LocalHook.h"
#include <chrono>
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

    int (*origin_launch_async_)(void *) = nullptr;
    int (*origin_launch_config_)(int, int, void *) = nullptr;
    int (*origin_launch_arg_set_)(const void *, size_t, size_t) = nullptr;
    // int (*origin_xpu_wait_)(void *) = nullptr;
    void *(*origin_dlsym_)(void *, const char *) = nullptr;
    void *(*origin_dlopen_)(const char *, int) = nullptr;
    int (*origin_print_)(const char*, ...)  = nullptr;
    int (*origin_fprintf_)(void* , const char*, ...)  = nullptr;
};

typedef utils::Singleton<CpuHookWrapper> SingletonCpuHookWrapper;

CpuHookWrapper::~CpuHookWrapper() {}

int CpuHookWrapper::local_launch_async(void *func) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    if (wrapper_instance->origin_launch_async_ != nullptr) {
        timer::record_time(/*ph=*/"B", /*name=*/__FUNCTION__, /*runtime api=*/"runtime api");
        int s = wrapper_instance->origin_launch_async_(func);
        timer::record_time(/*ph=*/"E", /*name=*/__FUNCTION__, /*runtime api=*/"runtime api");
        return s;
    }
    return 0;
}

int CpuHookWrapper::local_launch_config(int nclusters, int ncores,
                                        void *stream) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    if (wrapper_instance->origin_launch_config_ != nullptr) {
        return wrapper_instance->origin_launch_config_(nclusters, ncores,
                                                       stream);
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

int CpuHookWrapper::local_xpu_wait(void *stream) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    if (origin_xpu_wait_ != nullptr) {
        timer::record_time(/*ph=*/"B", /*name=*/"local_xpu_wait", /*runtime api=*/"runtime api", /*cname=*/"good");
        int status = origin_xpu_wait_(stream);
        timer::record_time(/*ph=*/"E", /*name=*/"local_xpu_wait", /*runtime api=*/"runtime api", /*cname=*/"good");
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

// REGISTER_LOCAL_HOOK(xpu_launch_async, (void *)local_launch_async, (void**)&Target_launch_async);
// REGISTER_LOCAL_HOOK(xpu_launch_async, (void *)CpuHookWrapper::local_launch_async,
//              (void **)&SingletonCpuHookWrapper::instance()
//                  .get_elem()
//                  ->origin_launch_async_);

REGISTERHOOK(xpu_launch_async, (void *)CpuHookWrapper::local_launch_async,
             (void **)&SingletonCpuHookWrapper::instance()
                 .get_elem()
                 ->origin_launch_async_);
// REGISTERHOOK(xpu_launch_config, (void *)CpuHookWrapper::local_launch_config,
//              (void **)&SingletonCpuHookWrapper::instance()
//                  .get_elem()
//                  ->origin_launch_config_);
// REGISTERHOOK(xpu_launch_argument_set,
//              (void *)CpuHookWrapper::local_launch_arg_set,
//              (void **)&SingletonCpuHookWrapper::instance()
//                  .get_elem()
//                  ->origin_launch_arg_set_);

REGISTERHOOK(
    xpu_wait, (void *)CpuHookWrapper::local_xpu_wait, (void **)&origin_xpu_wait_);

// REGISTERHOOK(
//     dlsym, (void *)CpuHookWrapper::local_dlsym,
//     (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_dlsym_);

// REGISTERHOOK(
//     dlopen, (void *)CpuHookWrapper::local_dlopen,
//     (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_dlopen_);

// REGISTERHOOK(printf, (void *)CpuHookWrapper::local_print,
//     (void**)&SingletonCpuHookWrapper::instance().get_elem()->origin_print_);

// REGISTER_LOCAL_HOOK(printf,  (void *)CpuHookWrapper::local_print, (void**)&SingletonCpuHookWrapper::instance().get_elem()->origin_print_);

REGISTERHOOK(fprintf, (void *)CpuHookWrapper::local_fprintf,
    (void**)&SingletonCpuHookWrapper::instance().get_elem()->origin_fprintf_);

namespace cpu_hook {
void register_cpu_hook() {
    // this function do nothing, but can not remove
}
} // namespace cpu_hook
