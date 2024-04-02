#ifndef LOCAL_HOOK_H
#define LOCAL_HOOK_H
#include <vector>
#include <string>

namespace local_hook {
void install_local_hook(void *hooked_func, void *payload_func,
                  void **trampoline_ptr) ;


struct LocalHookInfo {
    std::string symbol;
    void* new_func;
    void** trampoline;
    bool installed;
};

class LocalHookRegistrar {
public:
    LocalHookRegistrar() {}
    void install();
    void add(LocalHookInfo info);
private:
    std::vector<LocalHookInfo> hooks;
};

class LocalHookRegistration {
public:
    LocalHookRegistration(std::string symbol, void* new_func, void** trampoline);
};

void install_local_hooks();
}   // namespace local_hook

#define REGISTER_LOCAL_HOOK(symbol, payload, trampoline) \
    static local_hook::LocalHookRegistration registration##symbol(#symbol, payload, trampoline);

#endif