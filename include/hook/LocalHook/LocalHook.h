#ifndef LOCAL_HOOK_H
#define LOCAL_HOOK_H
namespace local_hook {
void install_local_hook(void *hooked_func, void *payload_func,
                  void **trampoline_ptr) ;
}   // namespace local_hook
#endif