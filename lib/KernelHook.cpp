#include "../include/KernelHook.h"
#include <functional>
#include <iostream>
#include <link.h>
#include <list>
#include <vector>
// #include "eager/framework/function_version.h"

namespace kernel_cache {

struct xpu_kernel {
    /// Combination of kernel place and type:
    /// [31:16] kernel place, KP_CPU or KP_XPU
    /// [15:0]  kernel type, KT_CLUSTER or KT_SDCDNN
    uint32_t type : 16;
    uint32_t place : 16;
    /// kernel code address on CPU Memory
    uint64_t code_addr;
    /// kernel code size in bytes
    uint32_t code_byte_size;
    /// initial program counter
    uint32_t code_pc;
    /// dword size kernel needed to transfer params
    /// essentially, this is the count of param registers needed
    uint32_t param_dword_size;
    /// kernel code hash, for cache indexing
    uint64_t hash;
    /// (maybe mangled) function name
    const char *name;
    /// private data structure used by xpu runtime
    void *rt_private;
};

template <typename TYPE> class Singleton {
  public:
    static Singleton *instance();
    Singleton() : elem(new TYPE()) {}
    ~Singleton() { delete elem; }
    TYPE *get_elem();

  private:
    TYPE *elem;
};

template <typename TYPE> Singleton<TYPE> *Singleton<TYPE>::instance() {
    static Singleton *inst = new Singleton();
    return inst;
}

template <typename TYPE> TYPE *Singleton<TYPE>::get_elem() { return elem; }

typedef Singleton<LaunchConfigParams> ConfigSingleton;
typedef Singleton<LaunchArgSetParams> ArgSetSingleton;
typedef Singleton<LaunchArgSetParamsList> ArgSetListSingleton;
typedef Singleton<LaunchKernelParams> LaunchAsyncSingleton;
typedef Singleton<GraphCacheEntry> GraphCacheEntrySingleton;

LaunchConfigParams::LaunchConfigParams(const LaunchConfigParams &rhs) {
    nclusters = rhs.nclusters;
    ncores = rhs.ncores;
    stream = rhs.stream;
}

LaunchArgSetParams::LaunchArgSetParams(const LaunchArgSetParams &rhs) {
    arg = rhs.arg;
    size = rhs.size;
    offset = rhs.offset;
}

LaunchKernelParams::LaunchKernelParams(const LaunchKernelParams &rhs) {
    func = rhs.func;
}

void OpCacheEntry::execute() {
    if (!kernel_cache_entry_list_) {
        return;
    }

    for (auto &kernel_cache_entry : *kernel_cache_entry_list_) {
        // xpu_launch_config
        // auto config_params = kernel_cache_entry.config_params_;
        // xpu_launch_config(config_params.nclusters, config_params.ncores,
        // config_params.stream);

        // // xpu_launch_argument_set
        // auto arg_set_params_list = kernel_cache_entry.arg_set_params_list_;
        // for (auto& arg_set_params : arg_set_params_list) {
        //     xpu_launch_argument_set(arg_set_params.arg, arg_set_params.size,
        //     arg_set_params.offset);
        // }

        // // xpu_launch_async
        // auto kernel_params = kernel_cache_entry.kernel_params_;
        // xpu_launch_async(kernel_params.func);
    }
}

void OpCacheEntry::add_entry(KernelCacheEntry &kernel_cache_entry) {
    kernel_cache_entry_list_->push_back(kernel_cache_entry);
}

int64_t OpCacheEntry::size() { return kernel_cache_entry_list_->size(); }

void GraphCacheEntry::execute() {
    // if (!kernel_cache_entry_list_) {
    //     return;
    // }

    // for (auto& kernel_cache_entry : *kernel_cache_entry_list_) {
    //     // xpu_launch_config
    //     auto config_params = kernel_cache_entry.config_params_;
    //     xpu_launch_config(config_params.nclusters, config_params.ncores,
    //     config_params.stream);

    //     // xpu_launch_argument_set
    //     auto arg_set_params_list = kernel_cache_entry.arg_set_params_list_;
    //     for (auto& arg_set_params : arg_set_params_list) {
    //         xpu_launch_argument_set(arg_set_params.arg, arg_set_params.size,
    //         arg_set_params.offset);
    //     }

    //     // xpu_launch_async
    //     auto kernel_params = kernel_cache_entry.kernel_params_;
    //     xpu_launch_async(kernel_params.func);
    // }
    if (!op_cache_entry_list_) {
        return;
    }
    for (auto &op_cache_entry : *op_cache_entry_list_) {
        op_cache_entry.execute();
    }
}

void GraphCacheEntry::add_entry(OpCacheEntry &op_cache_entry) {
    op_cache_entry_list_->push_back(op_cache_entry);
}

int64_t GraphCacheEntry::size() { return op_cache_entry_list_->size(); }

// KernelCache* KernelCache::instance = nullptr;

std::shared_ptr<GraphCacheEntry> KernelCache::get(int64_t key) {
    if (graph_entry_map_.find(key) == graph_entry_map_.end()) {
        return nullptr;
    }
    return graph_entry_map_[key];
}

void KernelCache::start_capture_launch_params(int64_t k) {
    key_ = k;
    capture_graph_ = true;
    graph_cache_entry_ = GraphCacheEntry();
}

void KernelCache::stop_capture_launch_params() {
    capture_graph_ = false;
    graph_entry_map_[key_] =
        std::make_shared<GraphCacheEntry>(graph_cache_entry_);
}

void KernelCache::start_capture_op(std::string name) {
    // op_name_ = name;
    capture_op_ = true;
    op_cache_entry_ = OpCacheEntry(name);
}

void KernelCache::stop_capture_op() {
    capture_op_ = false;
    if (op_cache_entry_.size() == 0) {
        return;
    }
    graph_cache_entry_.add_entry(op_cache_entry_);
}

GraphCacheEntry KernelCache::get_graph_cache_entry() {
    return graph_cache_entry_;
}

OpCacheEntry KernelCache::get_op_cache_entry() { return op_cache_entry_; }

void KernelCache::register_graph_cache_entry(int64_t key,
                                             GraphCacheEntry &entry) {
    if (graph_entry_map_.find(key) != graph_entry_map_.end()) {
        // check error
        return;
    }
    graph_entry_map_[key] = std::make_shared<GraphCacheEntry>(entry);
}

typedef Singleton<KernelCache> KernelCacheSingleton;

// typedef int (*LaunchAsyncFuncType)(void*);
// typedef int (*LaunchConfigFuncType)(int, int, void*);
// typedef int (*LaunchArgSetFuncType)(const void* arg, size_t, size_t);

class HookWrapper {
  public:
    HookWrapper() {}
    static int local_launch_async(void *func);
    static int local_launch_config(int nclusters, int ncores, void *stream);
    static int local_launch_arg_set(const void *arg, size_t size,
                                    size_t offset);
    static HookWrapper *instance();
    int (*origin_launch_async_)(void *){nullptr};
    int (*origin_launch_config_)(int, int, void *){nullptr};
    int (*origin_launch_arg_set_)(const void *, size_t, size_t){nullptr};
};

HookWrapper *HookWrapper::instance() {
    static HookWrapper *instance = new HookWrapper();
    return instance;
}

int HookWrapper::local_launch_async(void *func) {
    struct xpu_kernel *kernal = (xpu_kernel *)(func);
    std::string name = kernal->name;
    // std::cout << "kenrel name: " << name << std::endl;
    // KernelCache* cache = KernelCache::create();
    KernelCache *cache = KernelCacheSingleton::instance()->get_elem();

    if (cache->enable_capture()) {
        static ConfigSingleton *config_inst = ConfigSingleton::instance();
        LaunchConfigParams launch_config_params(*(config_inst->get_elem()));

        static ArgSetListSingleton *argset_list_inst =
            ArgSetListSingleton::instance();
        LaunchArgSetParamsList launch_arg_set_params_list(
            *argset_list_inst->get_elem());

        static LaunchAsyncSingleton *inst = LaunchAsyncSingleton::instance();
        LaunchKernelParams launch_params(func);

        OpCacheEntry op_cache_entry = cache->get_op_cache_entry();
        KernelCacheEntry kernel_cache_entry{
            launch_config_params, launch_arg_set_params_list, launch_params};
        op_cache_entry.add_entry(kernel_cache_entry);
        // std::cout << "cache_entry_size: " << op_cache_entry.size() <<
        // std::endl;
    }

    auto wrapper_instance = HookWrapper::instance();
    if (wrapper_instance->origin_launch_async_ != nullptr) {
        // std::cout << "execute origin launch" << std::endl;
        return wrapper_instance->origin_launch_async_(func);
    }
    return 0;
}

int HookWrapper::local_launch_config(int nclusters, int ncores, void *stream) {
    // KernelCache* cache = KernelCache::create();
    KernelCache *cache = KernelCacheSingleton::instance()->get_elem();
    if (cache->enable_capture()) {
        static ConfigSingleton *config_inst = ConfigSingleton::instance();
        LaunchConfigParams *config_params = config_inst->get_elem();
        config_params->nclusters = nclusters;
        config_params->ncores = ncores;
        config_params->stream = stream;
    }

    auto wrapper_instance = HookWrapper::instance();
    if (wrapper_instance->origin_launch_config_ != nullptr) {
        // std::cout << "execute origin launch config" << std::endl;
        return wrapper_instance->origin_launch_config_(nclusters, ncores,
                                                       stream);
    }
    return 0;
}

int HookWrapper::local_launch_arg_set(const void *arg, size_t size,
                                      size_t offset) {
    // KernelCache* cache = KernelCache::create();
    KernelCache *cache = KernelCacheSingleton::instance()->get_elem();
    if (cache->enable_capture()) {
        // std::cout << "capture launch arg set!!!!!!!!!!!!!!!!!" << std::endl;
        static ArgSetListSingleton *argset_list_inst =
            ArgSetListSingleton::instance();
        LaunchArgSetParamsList *arg_set_params_list =
            argset_list_inst->get_elem();
        char *arg_value = new char[size];
        memcpy((void *)arg_value, arg, size);
        arg_set_params_list->emplace_back(arg_value, size, offset);
    }
    // std::cout << "arg cpu ptr: " << arg << std::endl;
    // std::cout << "arg size: " << size << std::endl;
    // std::cout << "arg offset: " << offset << std::endl;
    auto wrapper_instance = HookWrapper::instance();
    if (wrapper_instance->origin_launch_arg_set_ != nullptr) {
        // std::cout << "execute origin launch arg set" << std::endl;
        return wrapper_instance->origin_launch_arg_set_(arg, size, offset);
    }
    return 0;
}

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

HookRegistrar::HookRegistrar() : hook_num_(0) {}

void HookRegistrar::register_hook(HookInfo hook) {
    hooks_.push_back(std::make_shared<HookInfo>(hook));
    hook_num_++;
}

HookList HookRegistrar::get_hooks() const { return hooks_; }

HookRegistrar *HookRegistrar::instance() {
    static HookRegistrar *inst = new HookRegistrar();
    return inst;
}

// struct CppHookInfo {
//     std::string func_str;
//     void* new_func;
//     void** origin_func;
// };

// typedef std::list<std::shared_ptr<CppHookInfo>> CppHookList;

// template<typename HookInfoType>
// class CppHookRegistrar {
// public:
//     typedef std::list<std::shared_ptr<HookInfoType>> HookListType;
//     CppHookRegistrar();
//     static CppHookRegistrar* instance();
//     void register_hook(HookInfoType hookinfo);
//     HookListType get_hooks() const;
//     void try_get_origin_func(std::string lib_name);
// private:
//     HookListType hooks_;
//     int64_t hook_num_;
// }

// template <typename HookInfoType>
// CppHookRegistrar<HookInfoType>::CppHookRegistrar() : hook_num_(0) {}

// template <typename HookInfoType>
// CppHookRegistrar<HookInfoType>*
// CppHookRegistrar<HookInfoType>::register_hook(HookInfoType hook) {
//     hooks_.push_back(std::make_shared<HookInfoType>(hook));
//     hook_num_++;
// }

// template <typename HookInfoType>
// CppHookRegistrar<HookInfoType>::HookListType
// CppHookRegistrar<HookInfoType>::get_hooks() const {
//     return hooks_;
// }

// template <typename HookInfoType>
// CppHookRegistrar<<HookInfoType>* CppHookRegistrar<HookInfoType>::instance() {
//     static CppHookRegistrar<HookInfoType>* inst = new
//     CppHookRegistrar<HookInfoType>(); return inst;
// }

// template <typename HookInfoType>
// void CppHookRegistrar<HookInfoType>::try_get_origin_func(std::string
// lib_name) {
//     // return
// }

void HookRegistrar::try_get_origin_func(std::string lib_name) {
    for (auto hook_ptr : hooks_) {
        if (*(hook_ptr->origin_func) == nullptr) {
            void *handle = dlopen(lib_name.c_str(), RTLD_LAZY);
            void *func_ptr = dlsym(handle, hook_ptr->sym_name.c_str());
            if (func_ptr) {
                *(hook_ptr->origin_func) = func_ptr;
                --hook_num_;
            }
            if (hook_num_ == 0) {
                break;
            }
        }
    }
}

typedef ElfW(Rela) RelaAddr;
typedef ElfW(Sym) SymAddr;

struct PltInfo {
    std::string lib_name;
    char *base_addr;
    int pltrelsz;
    char *dynstr;
    RelaAddr *rela_plt; // .rela.plt 段起始地址
    SymAddr *dynsym;    // .dynsym 段起始地址
};

typedef std::vector<PltInfo> PltInfoVec;

// int callback(struct dl_phdr_info* info, size_t size, void* data) {
//     // 待从 .dynamic 段中查找出的信息
//     const char* lib_name = info->dlpi_name;    // 动态库名称
//     char* base_addr = reinterpret_cast<char*>(info->dlpi_addr);
//     int pltrelsz = 0;             // .rela.plt 段大小
//     char* dynstr = NULL;          // .dynstr 段起始地址
//     RelaAddr* rela_plt = NULL;    // .rela.plt 段起始地址
//     SymAddr* dynsym = NULL;       // .dynsym 段起始地址

//     // std::cout << "lib_name: " << lib_name << std::endl;
//     // std::string lib_name_str =
//     //
//     "/ssd1/wangfangfei/projects/baidu/xpu/XMLIR/build/tools/torch_xmlir/python_packages/torch_xmlir/torch_xmlir/_XMLIRC.cpython-38-x86_64-linux-gnu.so";
//     // if (lib_name_str != std::string(lib_name)) {
//     //     return 0;
//     // }

//     HookRegistrar* reg = HookRegistrar::instance();
//     if (reg->get_hook_num() > 0) {
//         reg->try_get_origin_func(lib_name);
//     }

//     // 遍历当前动态库中所有段的信息
//     for (size_t i = 0; i < info->dlpi_phnum; i++) {
//         const ElfW(Phdr)* phdr = &info->dlpi_phdr[i];

//         // 如果不是 .dynamic 段则跳过
//         if (phdr->p_type != PT_DYNAMIC) {
//             continue;
//         }

//         int dynEntryCount = phdr->p_memsz / sizeof(ElfW(Dyn));
//         ElfW(Dyn)* dyn = (ElfW(Dyn)*)(phdr->p_vaddr + info->dlpi_addr);

//         // 遍历获取 .dynamic 段中的信息
//         for (int j = 0; j < dynEntryCount; j++) {
//             ElfW(Dyn)* entry = &dyn[j];
//             switch (dyn->d_tag) {
//             case DT_PLTRELSZ: {
//                 pltrelsz = dyn->d_un.d_val;
//                 break;
//             }
//             case DT_JMPREL: {
//                 rela_plt = (ElfW(Rela)*)(dyn->d_un.d_ptr);
//                 break;
//             }
//             case DT_STRTAB: {
//                 dynstr = (char*)(dyn->d_un.d_ptr);
//                 break;
//             }
//             case DT_SYMTAB: {
//                 dynsym = (ElfW(Sym)*)(dyn->d_un.d_ptr);
//                 break;
//             }
//             }
//             dyn++;
//         }
//     }

//     PltInfoVec* plt_info_vec = static_cast<PltInfoVec*>(data);
//     plt_info_vec->emplace_back(PltInfo{lib_name, base_addr, pltrelsz, dynstr,
//     rela_plt, dynsym});

//     return 0;
// }

PltInfoVec collect_plt() {
    PltInfoVec plt_info_vec;
    // dl_iterate_phdr(callback, (void*)&plt_info_vec);
    return plt_info_vec;
}

void install_hook() {
    static HookRegistrar *reg = HookRegistrar::instance();
    auto plt_info_vec = collect_plt();
    for (auto &plt_info : plt_info_vec) {
        int relaEntryCount = plt_info.pltrelsz / sizeof(ElfW(Rela));
        for (int i = 0; i < relaEntryCount; i++) {
            RelaAddr *entry = &(plt_info.rela_plt[i]);
            int r_sym = ELF64_R_SYM(entry->r_info);
            int st_name = plt_info.dynsym[r_sym].st_name;
            char *name = &plt_info.dynstr[st_name];
            // std::cout << "sym name: " << name << std::endl;

            for (auto hook_info : reg->get_hooks()) {
                if (std::string(name) == hook_info->sym_name) {
                    // std::cout << "found func: " << hook_info->sym_name <<
                    // std::endl;
                    uintptr_t hook_point =
                        (uintptr_t)(plt_info.base_addr + entry->r_offset);
                    *(void **)hook_point = (void *)hook_info->new_func;
                }
            }
        }
    }
}

class HookRegistration {
  public:
    HookRegistration(std::string name, void *new_func, void **old_func);
};

HookRegistration::HookRegistration(std::string name, void *new_func,
                                   void **old_func) {
    static HookRegistrar *reg = HookRegistrar::instance();
    reg->register_hook(HookInfo{name, new_func, old_func});
}

#define REGISTERHOOK(name, new_func, old_func)                                 \
    HookRegistration registration##name(#name, new_func, old_func);

// REGISTERHOOK(xpu_launch_async, (void*)HookWrapper::local_launch_async,
// (void**)&HookWrapper::instance()->origin_launch_async_);
// REGISTERHOOK(xpu_launch_config, (void*)HookWrapper::local_launch_config,
// (void**)&HookWrapper::instance()->origin_launch_config_); REGISTERHOOK(
//         xpu_launch_argument_set,
//         (void*)HookWrapper::local_launch_arg_set,
//         (void**)&HookWrapper::instance()->origin_launch_arg_set_);

void init_kernel_cache(pybind11::module &m) {
    m.def("install_hook", []() { install_hook(); });
}

} // namespace kernel_cache