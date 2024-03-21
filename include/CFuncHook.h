#ifndef KERNEL_HOOK_H
#define KERNEL_HOOK_H
#include <list>
#include <map>
#include <memory>
#include <pybind11/pybind11.h>
namespace kernel_hook {

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

void init_kernel_cache(pybind11::module &m);

void install_hook();

} // namespace kernel_hook

#define REGISTERHOOK(name, new_func, old_func)                                 \
    kernel_hook::HookRegistration registration##name(#name, new_func, old_func);

#endif