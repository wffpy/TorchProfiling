#include "hook/CFuncHook.h"
#include "utils/BackTrace.h"
#include "utils/Log/Log.h"
#include <functional>
#include <iostream>
#include <link.h>
#include <list>
#include <mutex>
#include <vector>

namespace cfunc_hook {
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
    std::string name_str = lib_name;
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
    LOG() << "hook num: " << reg->get_hook_num();
    auto plt_info_vec = collect_plt();
    for (auto &plt_info : plt_info_vec) {
        int relaEntryCount = plt_info.pltrelsz / sizeof(ElfW(Rela));
        for (int i = 0; i < relaEntryCount; i++) {
            RelaAddr *entry = &(plt_info.rela_plt[i]);
            int r_sym = ELF64_R_SYM(entry->r_info);
            int st_name = plt_info.dynsym[r_sym].st_name;
            char *name = &plt_info.dynstr[st_name];
            std::string lib_name = plt_info.lib_name;
            auto iter = lib_name.find("Hook");
            if (iter != std::string::npos) {
                continue;
            }
            for (auto hook_info : reg->get_hooks()) {
                if (std::string(name) == hook_info->sym_name) {
                    DLOG() << "found lib name: " << plt_info.lib_name;
                    DLOG() << "found func: " << hook_info->sym_name;
                    uintptr_t hook_point =
                        (uintptr_t)(plt_info.base_addr + entry->r_offset);
                    *(void **)hook_point = (void *)hook_info->new_func;
                }
            }
        }
    }
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
            if (func_ptr != nullptr) {
                DLOG() << "hooked func: " << hook_ptr->sym_name;
                DLOG() << "ptr: " << std::hex << func_ptr;
                *(hook_ptr->origin_func) = func_ptr;
                --hook_num_;
            }
            if (hook_num_ == 0) {
                break;
            }
        }
    }
}

HookRegistration::HookRegistration(std::string name, void *new_func,
                                   void **old_func) {
    static HookRegistrar *reg = HookRegistrar::instance();
    reg->register_hook(HookInfo{name, new_func, old_func});
}


int analysis_lib_name(struct dl_phdr_info *info, size_t size, void *data) {
    const char *lib_name = info->dlpi_name;
    auto vec = static_cast<std::vector<std::string>*>(data);
    vec->emplace_back(lib_name);
    return 0;
}
std::vector<std::string> get_libs() {
    std::vector<std::string> lib_vec;
    dl_iterate_phdr(analysis_lib_name, (void *)&lib_vec);
    return lib_vec;
}

} // namespace cfunc_hook
