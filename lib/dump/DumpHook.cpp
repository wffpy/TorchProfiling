#include "dump/dump.h"

#include "cpu/CpuHook.h"
#include "hook/CFuncHook.h"
#include "utils/Utils.h"
#include "utils/Log/Log.h"
#include "utils/Timer/Timer.h"
#include <stdarg.h>
#include <cxxabi.h>
#include <regex>
#include <iostream>
#include <sstream>
#include <fstream>

#ifdef XPU_DEV
#include "xpu/runtime.h"
#endif

using namespace cpu_hook;

int64_t get_rank_env() {
    const char* rank_str = std::getenv("RANK");
    if (rank_str == nullptr) {
        return 0;
    }
    int64_t rank = std::stoi(rank_str);
    return rank;
}

bool get_dump_env() {
    const char* dump_str = std::getenv("DEFAULT_DUMP");
    if (dump_str == nullptr) {
        return false;
    }
    if (strcmp(dump_str, "true") == 0 || strcmp(dump_str, "1") == 0) {
        return true;
    }
    return false;
}

int64_t get_cap() {
    const char* cap_str = std::getenv("LAUNCH_CAP_NUM");
    if (cap_str == nullptr) {
        return 10;
    }
    int64_t cap = std::stoi(cap_str);
    return cap;
}

struct TensorInfo {
    uint64_t ptr;
    int64_t elem_size;
};

typedef std::map<uint64_t, TensorInfo> TensorMap;

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
    void* rt_private;
};

struct ParsedType {
    std::string base_type;     // base type
    bool is_const = false;     // is const type
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

        if (token.find("const") != std::string::npos) {
            parsed_type.is_const = true;
        }

        if (token.find('*') != std::string::npos) {
            ++(parsed_type.prointer_level);
            if (token.size() == 1) {
                continue;
            }
        }

        if (token.find("float") != std::string::npos) {
            parsed_type.base_type = "float";
        } else if (token.find("int") != std::string::npos || token.find("int32_t") != std::string::npos) {
            parsed_type.base_type = "int";
        } else if (token.find("int64_t") != std::string::npos || token.find("long") != std::string::npos) {
            parsed_type.base_type = "int64_t";
        } else if (token.find("float16") != std::string::npos) {
            parsed_type.base_type = "float16";
        } else if (token.find("bfloat16") != std::string::npos) {
            parsed_type.base_type = "bfloat16";
        } else if (token.find("bool") != std::string::npos) {
            parsed_type.base_type = "bool";
        } else if (token.find("int8_t") != std::string::npos) {
            parsed_type.base_type = "int8_t";
        } else if (token.find("char") != std::string::npos) {
            parsed_type.base_type = "char";
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
    LaunchInfo() {
        params_.resize(128);
    }
    LaunchInfo(std::string func_name);
    ~LaunchInfo() {}
    std::string to_string(const TensorMap& tensor_map, const int64_t cap_index) const;
    void set_func_name(std::string func_name);
    void copy_args(const void* params, int64_t size, int64_t offset);

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

void LaunchInfo::copy_args(const void* params, int64_t size, int64_t offset) {
    void* data_ptr = params_.data();
    std::memmove(data_ptr + offset, params, size);
    size_ = size;
}

std::string LaunchInfo::to_string(const TensorMap& tensor_map, const int64_t cap_index) const {
    std::string str = "func_name: " + func_name_ + "\n";
    int status = 0;
    char* demangled = abi::__cxa_demangle(func_name_.c_str(), nullptr, nullptr, &status);
    CHECK(status == 0, "demangled name failed");
    auto param_types = extractParameterTypes(demangled);
    int64_t index = 0;
    int64_t pos = 0;
    for (auto type : param_types) {
        LOG() << "param type: " << type;
        auto parsed_type = parseType(type);
        DLOG() << "  base_type: " << parsed_type.base_type;
        DLOG() << "  is_const: " << parsed_type.is_const;
        DLOG() << "  prointer_level: " << parsed_type.prointer_level;
        // memory ptr
        if (parsed_type.prointer_level > 0) {
            char* cpu_ptr = ((char*)params_.data()) + pos;
            // input memory
            if (parsed_type.is_const) {
                LOG() << "input memory " << index << ": " << std::hex << *((int64_t*)cpu_ptr);
                uint64_t dev_ptr = *((uint64_t*)cpu_ptr);
                int64_t bytes = 0;
                auto iter = tensor_map.find(dev_ptr);
                if (iter != tensor_map.end()) {
                    bytes = iter->second.elem_size;
                } else {
                    LOG() << "not find data ptr info: " << std::hex << dev_ptr;
                }

                char* host_data = (char*)malloc(bytes);
                if (bytes > 0) {
                    char* host_data = (char*)malloc(bytes);
#ifdef XPU_DEV
                    xpu_memcpy(host_data, dev_ptr, bytes, XPU_DEVICE_TO_HOST);
#endif

                    int64_t rank = get_rank_env();
                    std::string param_file_name = func_name_ + "_cap_" + std::to_string(cap_index) + "_rank_" + std::to_string(rank)
                            + "_param_" + parsed_type.base_type + "_" + std::to_string(index) + ".bin";
                    std::ofstream file(param_file_name.c_str(), std::ios::binary);
                    if (file.is_open()) {
                        file.write(host_data, bytes);
                        file.close();
                        LOG() << "save param to file: " << param_file_name;
                    }
                    // if (parsed_type.base_type == "float") {
                    //     // TODO: print tensor
                    // } else if (parsed_type.base_type == "int" || parsed_type.base_type == "int32_t") {
                    // } else if (parsed_type.base_type == "int64_t" || parsed_type.base_type == "long") {
                    // } else if (parsed_type.base_type == "float16" || parsed_type.base_type == "bfloat16") {
                    // }

                    free(host_data);
                }
            } else {
                LOG() << "output memory " << index << ": " << std::hex << *((int64_t*)cpu_ptr);
            }
            pos = pos + 8;
        } else if (parsed_type.base_type == "int64_t" || parsed_type.base_type == "long") {
            char* dev_ptr = ((char*)params_.data()) + pos;
            LOG() << "input memory " << index << ": " << *((int64_t*)dev_ptr);
            pos = pos + 8;
        } else if (parsed_type.base_type == "int" || parsed_type.base_type == "int32_t" || parsed_type.base_type == "float") {
            if (parsed_type.base_type == "float") {
                char* dev_ptr = ((char*)params_.data()) + pos;
                LOG() << "input " << index << ": " << *((float*)dev_ptr);
            } else if (parsed_type.base_type == "int" || parsed_type.base_type == "int32_t") {
                char* dev_ptr = ((char*)params_.data()) + pos;
                LOG() << "input " << index << ": " << *((int32_t*)dev_ptr);
            }
            pos = pos + 4;
        } else if (parsed_type.base_type == "bfloat16" || parsed_type.base_type == "float16") {
            pos = pos + 2;
        } else if (parsed_type.base_type == "bool" || parsed_type.base_type == "int8_t") {
            pos = pos + 1;
        } else {
            ELOG() << "not support type: " << parsed_type.base_type;
        }
        index += 1;
    }

    return str;
}

class CircularQueue {
public:
    CircularQueue(int64_t cap = 1000);
    ~CircularQueue();
    void enqueue_params(const void* params, int64_t size, int64_t offset);
    void enqueue_name(std::string name);
    void flash();
    std::shared_ptr<LaunchInfo> getLaunchInfoPtr();
    void record_tensor(const uint64_t ptr, int64_t size);
    void set_print_flag(bool flag);

private:
    int64_t capacity;
    int64_t index_;
    std::vector<std::shared_ptr<LaunchInfo>> queue;
    std::mutex mtx_;
    TensorMap tensor_map_;
    bool print_flag_;
};

CircularQueue::CircularQueue(int64_t cap) : capacity(cap), index_(0), print_flag_(false) {
    capacity = get_cap();
    queue.resize(capacity);
    for (int64_t i = 0; i < capacity; ++i) {
        queue[i] = std::make_shared<LaunchInfo>();
    }
    print_flag_ = get_dump_env();
}

CircularQueue::~CircularQueue() {
    if (print_flag_) {
        flash();
    }
}

void CircularQueue::enqueue_params(const void* params, int64_t size, int64_t offset) {
    std::lock_guard<std::mutex> lock(mtx_);
    int64_t r_index = index_ % capacity;
    queue[r_index]->copy_args(params, size, offset);
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
        LOG() << "dump launch info: " << real_cap - i;
        LOG() << info->to_string(tensor_map_, real_cap - i);
    }
}

void CircularQueue::record_tensor(const uint64_t ptr, int64_t size) {
    TensorInfo info = {ptr, size};
    tensor_map_[ptr] = info;
}

void CircularQueue::set_print_flag(bool flag) {
    print_flag_ = flag;
}

typedef utils::Singleton<CircularQueue> SingletonCircularQueue;

namespace dump {
void record_tensor(const uint64_t ptr, int64_t size) {
    SingletonCircularQueue::instance().get_elem()->record_tensor(ptr, size);
}
}    // namespace dump

class DumpHookWrapper {
public:
    DumpHookWrapper() {}
    ~DumpHookWrapper() {}
    static int local_launch_async(void* func);
    static int local_launch_config(int nclusters, int ncores, void* stream);
    static int local_launch_arg_set(const void* arg, size_t size, size_t offset);

    int (*origin_launch_async_)(void*) = nullptr;
    int (*origin_launch_config_)(int, int, void*) = nullptr;
    int (*origin_launch_arg_set_)(const void*, size_t, size_t) = nullptr;
};

typedef utils::Singleton<DumpHookWrapper> SingletonDumpHookWrapper;

int DumpHookWrapper::local_launch_async(void* func) {
    auto xpu_func = static_cast<xpu_kernel*>(func);
    auto cqueue = SingletonCircularQueue::instance().get_elem();
    const char* mangled_name = xpu_func->name;
    DLOG() << "local launch async begin: " << mangled_name;

    cqueue->enqueue_name(mangled_name);

    int s = 0;
    auto wrapper_instance = SingletonDumpHookWrapper::instance().get_elem();
    if (wrapper_instance->origin_launch_async_ != nullptr) {
        s = wrapper_instance->origin_launch_async_(func);
        if (s != 0) {
            SingletonCircularQueue::instance().get_elem()->set_print_flag(true);
        }
    }
    DLOG() << "local launch async end: " << mangled_name;
    return s;
}

REGISTERHOOK(
        cfunc_hook::HookType::kDUMP,
        xpu_launch_async,
        (void*)DumpHookWrapper::local_launch_async,
        (void**)&SingletonDumpHookWrapper::instance().get_elem()->origin_launch_async_);

int DumpHookWrapper::local_launch_arg_set(const void* arg, size_t size, size_t offset) {
    DLOG() << "launch arg set: size: " << size << " offset: " << offset << "\n";
    auto cqueue = SingletonCircularQueue::instance().get_elem();
    cqueue->enqueue_params(const_cast<void*>(arg), size, offset);

    int s = 0;
    auto wrapper_instance = SingletonDumpHookWrapper::instance().get_elem();
    if (wrapper_instance->origin_launch_arg_set_ != nullptr) {
        s = wrapper_instance->origin_launch_arg_set_(arg, size, offset);
    }
    return s;
}

REGISTERHOOK(
        cfunc_hook::HookType::kDUMP,
        xpu_launch_argument_set,
        (void*)DumpHookWrapper::local_launch_arg_set,
        (void**)&SingletonDumpHookWrapper::instance().get_elem()->origin_launch_arg_set_);
