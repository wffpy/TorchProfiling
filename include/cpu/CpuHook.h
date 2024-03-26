#ifndef CPU_HOOK_H
#define CPU_HOOK_H
#include <list>
#include <map>
#include <memory>
#include <mutex>
namespace cpu_hook {

// class CpuHookWrapper {
// public:
//     CpuHookWrapper() {}
//     ~CpuHookWrapper();
//     static int local_launch_async(void *func);
//     static int local_launch_config(int nclusters, int ncores, void *stream);
//     static int local_launch_arg_set(const void *arg, size_t size,
//                                     size_t offset);
//     static int local_xpu_wait(void* stream);

//     int (*origin_launch_async_)(void *){nullptr};
//     int (*origin_launch_config_)(int, int, void *){nullptr};
//     int (*origin_launch_arg_set_)(const void *, size_t, size_t){nullptr};
//     int (*origin_xpu_wait_)(void*){nullptr};
// };

// typedef utils::Singleton<CpuHookWrapper> SingletonCpuHookWrapper;

void register_cpu_hook();

}   // namespace cpu_hook
#endif