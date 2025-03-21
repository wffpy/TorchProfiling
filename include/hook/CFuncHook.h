#ifndef CFUNC_HOOK_H
#define CFUNC_HOOK_H

#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <string>

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace cfunc_hook {

enum class HookType {
    kNONE = 0,
    kDUMP,
    kPROFILE,
    kACCUMULATE_KERNEL_TIME,
    kDEBUG,
};

struct HookInfo {
    std::string sym_name;
    void *new_func;
    void **origin_func;
};

typedef std::list<std::shared_ptr<HookInfo>> HookList;
typedef std::map<HookType, HookList> NamedHookList;

class HookRegistrar {
  public:
    HookRegistrar();
    ~HookRegistrar();
    static HookRegistrar *instance();
    void register_hook(const HookType& category, HookInfo hookinfo);
    void set_current_category(HookType category);
    const HookList get_hooks();
    void try_get_origin_func(std::string lib_name);
    int64_t get_hook_num() { return hook_num_; }
    void *get_origin_func(std::string sym_name);

  private:
    HookType current_category_;
    NamedHookList hooks_;
    int64_t hook_num_;
    int64_t get_hook_category();
    std::map<std::string, void **> origin_func_map_;
    std::mutex mtx_;
};

class HookRegistration {
  public:
    HookRegistration(HookType category, std::string name, void *new_func, void **old_func);
};

std::vector<std::string> get_libs();

void install_got_hook(HookType category);

void register_got_hook(HookType category, std::string sym, void *new_func);

int64_t get_origin_func(std::string sym);

} // namespace cfunc_hook

#define REGISTERHOOK(category, category_name, name, new_func, old_func) \
    static cfunc_hook::HookRegistration __attribute__((used))           \
        registration_##category_name##_##name(category, #name, new_func, old_func);

#endif
