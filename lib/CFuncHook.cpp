#include "CFuncHook.h"
#include <functional>
#include <iostream>
#include <link.h>
#include <list>
#include <vector>
// #include "eager/framework/function_version.h"

namespace kernel_hook {

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

    auto wrapper_instance = HookWrapper::instance();
    if (wrapper_instance->origin_launch_async_ != nullptr) {
        // std::cout << "execute origin launch" << std::endl;
        return wrapper_instance->origin_launch_async_(func);
    }
    return 0;
}

int HookWrapper::local_launch_config(int nclusters, int ncores, void *stream) {
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
    auto wrapper_instance = HookWrapper::instance();
    if (wrapper_instance->origin_launch_arg_set_ != nullptr) {
        // std::cout << "execute origin launch arg set" << std::endl;
        return wrapper_instance->origin_launch_arg_set_(arg, size, offset);
    }
    return 0;
}

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

int callback(struct dl_phdr_info *info, size_t size, void *data) {
    // 待从 .dynamic 段中查找出的信息
    const char *lib_name = info->dlpi_name; // 动态库名称
    char *base_addr = reinterpret_cast<char *>(info->dlpi_addr);
    int pltrelsz = 0;          // .rela.plt 段大小
    char *dynstr = NULL;       // .dynstr 段起始地址
    RelaAddr *rela_plt = NULL; // .rela.plt 段起始地址
    SymAddr *dynsym = NULL;    // .dynsym 段起始地址

    HookRegistrar *reg = HookRegistrar::instance();
    if (reg->get_hook_num() > 0) {
        reg->try_get_origin_func(lib_name);
    }

    // 遍历当前动态库中所有段的信息
    for (size_t i = 0; i < info->dlpi_phnum; i++) {
        const ElfW(Phdr) *phdr = &info->dlpi_phdr[i];

        // 如果不是 .dynamic 段则跳过
        if (phdr->p_type != PT_DYNAMIC) {
            continue;
        }

        int dynEntryCount = phdr->p_memsz / sizeof(ElfW(Dyn));
        ElfW(Dyn) *dyn = (ElfW(Dyn) *)(phdr->p_vaddr + info->dlpi_addr);

        // 遍历获取 .dynamic 段中的信息
        for (int j = 0; j < dynEntryCount; j++) {
            ElfW(Dyn) *entry = &dyn[j];
            switch (dyn->d_tag) {
            case DT_PLTRELSZ: {
                pltrelsz = dyn->d_un.d_val;
                break;
            }
            case DT_JMPREL: {
                rela_plt = (ElfW(Rela) *)(dyn->d_un.d_ptr);
                break;
            }
            case DT_STRTAB: {
                dynstr = (char *)(dyn->d_un.d_ptr);
                break;
            }
            case DT_SYMTAB: {
                dynsym = (ElfW(Sym) *)(dyn->d_un.d_ptr);
                break;
            }
            }
            dyn++;
        }
    }

    PltInfoVec *plt_info_vec = static_cast<PltInfoVec *>(data);
    plt_info_vec->emplace_back(
        PltInfo{lib_name, base_addr, pltrelsz, dynstr, rela_plt, dynsym});

    return 0;
}

PltInfoVec collect_plt() {
    PltInfoVec plt_info_vec;
    dl_iterate_phdr(callback, (void *)&plt_info_vec);
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
            std::cout << "sym name: " << name << std::endl;

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

REGISTERHOOK(xpu_launch_async, (void *)HookWrapper::local_launch_async,
             (void **)&HookWrapper::instance()->origin_launch_async_);
REGISTERHOOK(xpu_launch_config, (void *)HookWrapper::local_launch_config,
             (void **)&HookWrapper::instance()->origin_launch_config_);
REGISTERHOOK(xpu_launch_argument_set, (void *)HookWrapper::local_launch_arg_set,
             (void **)&HookWrapper::instance()->origin_launch_arg_set_);

void init_kernel_cache(pybind11::module &m) {
    m.def("install_hook", []() { install_hook(); });
}

} // namespace kernel_hook