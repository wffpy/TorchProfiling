#include "cpu/CpuHook.h"
#include "hook/CFuncHook.h"
#include "utils/BackTrace.h"
#include "utils/Utils.h"
#include "utils/Log/Log.h"
#include "hook/LocalHook/LocalHook.h"
#include "utils/Timer/Timer.h"
#include "utils/Recorder/Recorder.h"
#include <stdarg.h>
#include <cxxabi.h>
#include <regex>
#include <iostream>
#include <sstream>


using namespace cpu_hook;
using namespace std::chrono;

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
    const char* name;
    /// private data structure used by xpu runtime
    void *rt_private;
};

struct ParsedType {
    std::string base_type;   // base type
    bool is_const = false;  // is const type
    int prointer_level = 0;    // pointer level float* -> 1, float** -> 2
};

std::vector<std::string> splitBySpace(const std::string& str) {
    std::vector<std::string> tokens;
    std::istringstream stream(str);
    std::string token;

    while (stream >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

ParsedType parseType(const std::string& typeStr) {
    ParsedType parsed_type;
    std::vector<std::string> tokens = splitBySpace(typeStr);
    for (auto token : tokens) {
        if (token == "const") {
            parsed_type.is_const = true;
            continue;
        }

        if (token.find('*') != std::string::npos) {
            ++(parsed_type.prointer_level);
            if (token.size() == 1) {
                continue;
            }
        }

        if (token.find('float') != std::string::npos) {
            parsed_type.base_type = "float";
        } else if (token.find('int') != std::string::npos || token.find('int32_t') != std::string::npos) {
            parsed_type.base_type = "int";
        } else if (token.find('int64_t') != std::string::npos || token.find('long') != std::string::npos) {
            parsed_type.base_type = "int64_t";
        } else if (token.find('float16') != std::string::npos) {
            parsed_type.base_type = "float16";
        } else if (token.find('bfloat16') != std::string::npos) {
            parsed_type.base_type = "bfloat16";
        } else if (token.find('bool') != std::string::npos) {
            parsed_type.base_type = "bool";
        } else if (token.find('int8_t') != std::string::npos) {
            parsed_type.base_type = "int8_t";
        }
    }
    return parsed_type;
}

std::vector<std::string> extractParameterTypes(const std::string& demangled) {
    std::vector<std::string> params;
    std::size_t start = demangled.find('(');
    std::size_t end = demangled.find(')');
    
    if (start != std::string::npos && end != std::string::npos && start < end) {
        std::string param_str = demangled.substr(start + 1, end - start - 1);
        if (!param_str.empty()) {
            // 分割参数字符串，支持复杂类型
            std::regex param_regex(R"(([^,]+(, )?))");
            auto begin = std::sregex_iterator(param_str.begin(), param_str.end(), param_regex);
            auto end = std::sregex_iterator();

            for (auto it = begin; it != end; ++it) {
                params.push_back(it->str());
            }
        }
    }
    return params;
}

class LaunchInfo {
public:
    LaunchInfo() { params_.resize(128); }
    LaunchInfo(std::string func_name);
    ~LaunchInfo() {}
    // void add_arg(void* arg);
    std::string to_string() const;
    void set_func_name(std::string func_name);
    void copy_args(const void* params, int64_t size);
private:
    int64_t size_;
    std::string func_name_;
    std::vector<uint32_t> params_;
};

LaunchInfo::LaunchInfo(std::string func_name) : func_name_(func_name) {
    params_.resize(128);
}

void LaunchInfo::set_func_name(std::string func_name) {
    func_name_ = func_name;
}

void LaunchInfo::copy_args(const void* params, int64_t size) {
    void* data_ptr = params_.data();
    std:memmove(data_ptr, params, size);
    size_ = size;
}

std::string LaunchInfo::to_string() const {
    std::string str = "func_name: " + func_name_ + "\n";
    int status = 0;
    char* demangled = abi::__cxa_demangle(func_name_.c_str(), nullptr, nullptr, &status);
    CHECK(status == 0, "demangled name failed");
    auto praam_types = extractParameterTypes(demangled);
    for (auto type : praam_types) {
        LOG() << "param type: " << type;
        auto parsed_type = parseType(type);
    }
    return str;
}

class CircularQueue {
public:
    CircularQueue(int64_t cap = 6);
    ~CircularQueue();
    void enqueue_params(const void* params, int64_t size);
    void enqueue_name(std::string name);
    void flash();
    std::shared_ptr<LaunchInfo> getLaunchInfoPtr();
private:
    int64_t capacity;
    int64_t index_;
    std::vector<std::shared_ptr<LaunchInfo>> queue;
    std::mutex mtx_;
};

CircularQueue::CircularQueue(int64_t cap) : capacity(cap), index_(0) {
    queue.resize(capacity);
    for (int64_t i = 0; i < capacity; ++i) {
        queue[i] = std::make_shared<LaunchInfo>();
    }
}

CircularQueue::~CircularQueue() { flash(); }

void CircularQueue::enqueue_params(const void* params, int64_t size) {
    std::lock_guard<std::mutex> lock(mtx_);
    int64_t r_index = index_ % capacity;
    queue[r_index]->copy_args(params, size);
}

void CircularQueue::enqueue_name(std::string name) {
    std::lock_guard<std::mutex> lock(mtx_);
    int64_t r_index = index_ % capacity;
    queue[r_index]->set_func_name(name);
    ++index_;
}

std::shared_ptr<LaunchInfo> CircularQueue::getLaunchInfoPtr() {
    int64_t r_index = index_ % capacity;
    return queue[r_index];
}

void CircularQueue::flash() {
    int64_t r_index = index_;
    int64_t real_cap = index_ > capacity ? capacity : index_;
    for (int64_t i = 0; i < real_cap; ++i) {
        int64_t r_index = (index_ - i - 1) % capacity;
        auto info = queue[r_index];
        LOG() << "dump launch info: " << r_index;
        LOG() << info->to_string();
    }
}

typedef utils::Singleton<CircularQueue> SingletonCircularQueue;

class DumpHookWrapper {
public:
    DumpHookWrapper() {}
    ~DumpHookWrapper() {}
    static int local_launch_async(void *func);
    static int local_launch_config(int nclusters, int ncores, void *stream);
    static int local_launch_arg_set(const void *arg, size_t size,
                                    size_t offset);

    int (*origin_launch_async_)(void *) = nullptr;
    int (*origin_launch_config_)(int, int, void *) = nullptr;
    int (*origin_launch_arg_set_)(const void *, size_t, size_t) = nullptr;
};

typedef utils::Singleton<DumpHookWrapper> SingletonDumpHookWrapper;

int DumpHookWrapper::local_launch_async(void *func) {
    auto xpu_func = static_cast<xpu_kernel*>(func);
    auto cqueue = SingletonCircularQueue::instance().get_elem();
    const char* mangled_name = xpu_func->name;
    LOG() << "mangled name: " << mangled_name;

    cqueue->enqueue_name(mangled_name);

    auto wrapper_instance = SingletonDumpHookWrapper::instance().get_elem();
    if (wrapper_instance->origin_launch_async_ != nullptr) {
        int s = wrapper_instance->origin_launch_async_(func);
        return s;
    }
    return 0;
}

REGISTERHOOK(cfunc_hook::HookType::kDUMP, xpu_launch_async, (void *)DumpHookWrapper::local_launch_async,
             (void **)&SingletonDumpHookWrapper::instance()
                 .get_elem()
                 ->origin_launch_async_);


int DumpHookWrapper::local_launch_arg_set(const void *arg, size_t size,
                                         size_t offset) {
    
    auto cqueue = SingletonCircularQueue::instance().get_elem();
    cqueue->enqueue_params(const_cast<void *>(arg), size);

    auto wrapper_instance = SingletonDumpHookWrapper::instance().get_elem();
    if (wrapper_instance->origin_launch_arg_set_ != nullptr) {
        return wrapper_instance->origin_launch_arg_set_(arg, size, offset);
    }
    return 0;
}

REGISTERHOOK(cfunc_hook::HookType::kDUMP, xpu_launch_argument_set, (void *)DumpHookWrapper::local_launch_arg_set,
             (void **)&SingletonDumpHookWrapper::instance()
                 .get_elem()
                 ->origin_launch_arg_set_);

namespace cpu_hook {
void register_dump_hook() {}
}   // namespace cpu_hook