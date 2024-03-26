#ifndef CFUNC_HOOK_H
#define CFUNC_HOOK_H

#include <list>
#include <map>
#include <memory>
#include <mutex>
namespace cfunc_hook {

struct HookInfo {
    std::string sym_name;
    void *new_func;
    void **origin_func;
};

typedef std::list<std::shared_ptr<HookInfo>> HookList;

class HookRegistrar {
  public:
    HookRegistrar();
    static HookRegistrar *instance();
    void register_hook(HookInfo hookinfo);
    HookList get_hooks() const;
    void try_get_origin_func(std::string lib_name);
    int64_t get_hook_num() { return hook_num_; }

  private:
    HookList hooks_;
    int64_t hook_num_;
};

class HookRegistration {
  public:
    HookRegistration(std::string name, void *new_func, void **old_func);
};

// class HookWrapper {
// public:
//     HookWrapper() {}
//     ~HookWrapper();
//     static int local_launch_async(void *func);
//     static int local_launch_config(int nclusters, int ncores, void *stream);
//     static int local_launch_arg_set(const void *arg, size_t size,
//                                     size_t offset);
//     static int local_xpu_wait(void* stream);

//     static HookWrapper *instance();
//     int (*origin_launch_async_)(void *){nullptr};
//     int (*origin_launch_config_)(int, int, void *){nullptr};
//     int (*origin_launch_arg_set_)(const void *, size_t, size_t){nullptr};
//     int (*origin_xpu_wait_)(void*){nullptr};
// private:
//     static std::once_flag flag;
//     static HookWrapper* inst;
// };

void install_hook();

} // namespace cfunc_hook

#define REGISTERHOOK(name, new_func, old_func)                                 \
    cfunc_hook::HookRegistration __attribute__((used)) registration##name(#name, new_func, old_func);

#endif