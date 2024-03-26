// #include "BackTrace.h"
// #include "CFuncHook.h"
// #include "cuda/GpuProfiler.h"
// #include <functional>
// #include <iostream>
// #include <link.h>
// #include <list>
// #include <mutex>
// #include <vector>

// using namespace kernel_hook;
// using namespace gpu_profiler;
// using namespace cpu_profiler;

// std::once_flag CpuHookWrapper::flag;
// CpuHookWrapper *CpuHookWrapper::inst = nullptr;

// CpuHookWrapper::~CpuHookWrapper() { delete inst; }

// CpuHookWrapper *SingletonCpuHookWrapper::instance()  {
//   std::call_once(flag, []() { inst = new CpuHookWrapper(); });
//   return inst;
// }

// int CpuHookWrapper::local_launch_async(void *func) {
//   trace::Tracer tracer(__FUNCTION__);
//   auto wrapper_instance = SingletonCpuHookWrapper::instance() ;
//   if (wrapper_instance->origin_launch_async_ != nullptr) {
//     // std::cout << "execute origin launch" << std::endl;
//     return wrapper_instance->origin_launch_async_(func);
//   }
//   return 0;
// }

// int CpuHookWrapper::local_launch_config(int nclusters, int ncores, void
// *stream) {
//   trace::Tracer tracer(__FUNCTION__);
//   auto wrapper_instance = SingletonCpuHookWrapper::instance() ;
//   if (wrapper_instance->origin_launch_config_ != nullptr) {
//     // std::cout << "execute origin launch config" << std::endl;
//     return wrapper_instance->origin_launch_config_(nclusters, ncores,
//     stream);
//   }
//   return 0;
// }

// int CpuHookWrapper::local_launch_arg_set(const void *arg, size_t size,
//                                       size_t offset) {
//   trace::Tracer tracer(__FUNCTION__);
//   auto wrapper_instance = SingletonCpuHookWrapper::instance() ;
//   if (wrapper_instance->origin_launch_arg_set_ != nullptr) {
//     // std::cout << "execute origin launch arg set" << std::endl;
//     return wrapper_instance->origin_launch_arg_set_(arg, size, offset);
//   }
//   return 0;
// }

// int CpuHookWrapper::local_xpu_wait(void *stream) {
//   trace::Tracer tracer(__FUNCTION__);
//   auto wrapper_instance = SingletonCpuHookWrapper::instance() ;
//   if (wrapper_instance->origin_xpu_wait_ != nullptr) {
//     // std::cout << "execute origin xpu wait" << std::endl;
//     return wrapper_instance->origin_xpu_wait_(stream);
//   } else {
//     std::cout << "origin xpu wait is null!!!!!!!!!!!!" << std::endl;
//     exit(0);
//   }
//   return 0;
// }

// HookRegistrar::HookRegistrar() : hook_num_(0) {}

// void HookRegistrar::register_hook(HookInfo hook) {
//   hooks_.push_back(std::make_shared<HookInfo>(hook));
//   hook_num_++;
// }

// HookList HookRegistrar::get_hooks() const { return hooks_; }

// HookRegistrar *HookRegistrar::instance() {
//   static HookRegistrar *inst = new HookRegistrar();
//   return inst;
// }

// void HookRegistrar::try_get_origin_func(std::string lib_name) {
//   for (auto hook_ptr : hooks_) {
//     if (*(hook_ptr->origin_func) == nullptr) {
//       void *handle = dlopen(lib_name.c_str(), RTLD_LAZY);
//       void *func_ptr = dlsym(handle, hook_ptr->sym_name.c_str());
//       if (func_ptr) {
//         *(hook_ptr->origin_func) = func_ptr;
//         --hook_num_;
//       }
//       if (hook_num_ == 0) {
//         break;
//       }
//     }
//   }
// }

// HookRegistration::HookRegistration(std::string name, void *new_func,
//                                    void **old_func) {
//   static HookRegistrar *reg = HookRegistrar::instance();
//   reg->register_hook(HookInfo{name, new_func, old_func});
// }

// REGISTERHOOK(xpu_launch_async, (void *)CpuHookWrapper::local_launch_async,
//              (void **)&SingletonCpuHookWrapper::instance()
//              ->origin_launch_async_);
// REGISTERHOOK(xpu_launch_config, (void *)CpuHookWrapper::local_launch_config,
//              (void **)&SingletonCpuHookWrapper::instance()
//              ->origin_launch_config_);
// REGISTERHOOK(xpu_launch_argument_set, (void
// *)CpuHookWrapper::local_launch_arg_set,
//              (void **)&SingletonCpuHookWrapper::instance()
//              ->origin_launch_arg_set_);

// REGISTERHOOK(xpu_wait, (void *)CpuHookWrapper::local_xpu_wait,
//              (void **)&SingletonCpuHookWrapper::instance()
//              ->origin_xpu_wait_);

// REGISTERHOOK(cudaLaunchKernel, (void
// *)GpuHookWrapper::local_cuda_launch_kernel,
//              (void **)&SingletonGpuHookWrapper::instance()
//                  .get_elem()
//                  ->oriign_cuda_launch_kernel_);
