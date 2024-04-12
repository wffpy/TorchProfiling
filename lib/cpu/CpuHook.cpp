#include "cpu/CpuHook.h"
#include "hook/CFuncHook.h"
#include "utils/BackTrace.h"
#include "utils/Utils.h"
#include "utils/Log/Log.h"
#include "hook/LocalHook/LocalHook.h"
using namespace cpu_hook;

class CpuHookWrapper {
  public:
    CpuHookWrapper() {}
    ~CpuHookWrapper();
    static int local_launch_async(void *func);
    static int local_launch_config(int nclusters, int ncores, void *stream);
    static int local_launch_arg_set(const void *arg, size_t size,
                                    size_t offset);
    static int local_xpu_wait(void *stream);
    static void *local_dlsym(void *handle, const char *symbol);;
    static void *local_dlopen(const char *filename, int flag);

    int (*origin_launch_async_)(void *){nullptr};
    int (*origin_launch_config_)(int, int, void *){nullptr};
    int (*origin_launch_arg_set_)(const void *, size_t, size_t){nullptr};
    int (*origin_xpu_wait_)(void *){nullptr};
    void *(*origin_dlsym_)(void *, const char *);;
    void *(*origin_dlopen_)(const char *, int);
};

typedef utils::Singleton<CpuHookWrapper> SingletonCpuHookWrapper;

CpuHookWrapper::~CpuHookWrapper() {}

int CpuHookWrapper::local_launch_async(void *func) {
    trace::Tracer tracer(__FUNCTION__);
    auto wrapper_instance = SingletonCpuHookWrapper::instance().get_elem();
    if (wrapper_instance->origin_launch_async_ != nullptr) {
        return wrapper_instance->origin_launch_async_(func);
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
    if (wrapper_instance->origin_xpu_wait_ != nullptr) {
        return wrapper_instance->origin_xpu_wait_(stream);
    } else {
        ELOG() << "origin xpu wait is null";
        exit(0);
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

REGISTER_LOCAL_HOOK(xpu_launch_async, (void *)local_launch_async, (void**)&Target_launch_async);
// REGISTER_LOCAL_HOOK(xpu_launch_async, (void *)CpuHookWrapper::local_launch_async,
//              (void **)&SingletonCpuHookWrapper::instance()
//                  .get_elem()
//                  ->origin_launch_async_);

// REGISTERHOOK(xpu_launch_async, (void *)CpuHookWrapper::local_launch_async,
//              (void **)&SingletonCpuHookWrapper::instance()
//                  .get_elem()
//                  ->origin_launch_async_);
// REGISTERHOOK(xpu_launch_config, (void *)CpuHookWrapper::local_launch_config,
//              (void **)&SingletonCpuHookWrapper::instance()
//                  .get_elem()
//                  ->origin_launch_config_);
// REGISTERHOOK(xpu_launch_argument_set,
//              (void *)CpuHookWrapper::local_launch_arg_set,
//              (void **)&SingletonCpuHookWrapper::instance()
//                  .get_elem()
//                  ->origin_launch_arg_set_);

// REGISTERHOOK(
//     xpu_wait, (void *)CpuHookWrapper::local_xpu_wait,
//     (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_xpu_wait_);
REGISTERHOOK(
    dlsym, (void *)CpuHookWrapper::local_dlsym,
    (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_dlsym_);

REGISTERHOOK(
    dlopen, (void *)CpuHookWrapper::local_dlopen,
    (void **)&SingletonCpuHookWrapper::instance().get_elem()->origin_dlopen_);

namespace cpu_hook {
void register_cpu_hook() {
    // this function do nothing, but can not remove
}
} // namespace cpu_hook
