#include "utils/BackTrace.h"
#include "utils/Utils.h"
#include <cstdlib>
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <iostream>
#include <mutex>
#include <regex>

using namespace trace;

class Recorder {
  public:
    // key: library name, value: map of function names and call times
    typedef std::map<std::string, std::map<std::string, int64_t>> RecorderMapT;
    Recorder() {}
    ~Recorder();
    void record(std::string lib_name, std::string func_name);
    static bool enable;

  private:
    RecorderMapT recorder_map;
    std::mutex mtx;
};

bool Recorder::enable = true;

void Recorder::record(std::string lib_name, std::string func_name) {
    std::unique_lock<std::mutex> lock(mtx);
    auto it = recorder_map.find(lib_name);
    if (it == recorder_map.end()) {
        recorder_map[lib_name] = std::map<std::string, int64_t>();
        recorder_map[lib_name][func_name] = 1;
        return;
    }
    auto func_iter = recorder_map[lib_name].find(func_name);
    if (func_iter == recorder_map[lib_name].end()) {
        recorder_map[lib_name][func_name] = 1;
        return;
    }
    func_iter->second++;
}

Recorder::~Recorder() {
    for (auto lib_iter = recorder_map.begin(); lib_iter != recorder_map.end();
         ++lib_iter) {
        std::string lib_name = lib_iter->first;
        utils::print_table_name(lib_name);
        for (auto func_iter = lib_iter->second.begin();
             func_iter != lib_iter->second.end(); ++func_iter) {
            utils::print_line(
                {func_iter->first, std::to_string(func_iter->second)});
            std::cout << std::endl;
        }
    }
}

typedef utils::Singleton<Recorder> RecorderSingleton;

// get the lib name from path
std::string get_file_from_path(const std::string path) {
    std::string lib_name = "";
    std::regex re("[^/\\\\]+$");
    std::smatch match;
    if (std::regex_search(path, match, re)) {
        lib_name = match.str();
    }
    return lib_name;
}

std::string demangle(const char *mangled_name) {
    int status = 0;
    char *demangled =
        abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);
    if (status == 0) {
        std::string result(demangled);
        std::free(demangled);
        return result;
    } else {
        return std::string(mangled_name);
    }
}

/*******************************************************************************
 *  Tracer
 * - get the backtrace of the current function
 *  - backtrace()
 *  - backtrace_symbols()
 * - count the call times of the current function
 *******************************************************************************/
Tracer::Tracer(std::string name)
    : max_depth(100), real_size(0), func_name(name) {
    stack.reserve(max_depth);
    trace();
    const char *print_backtrace = std::getenv("PRINT_BACKTRACE");
    if (print_backtrace &&
        (print_backtrace == "true" || print_backtrace == "TRUE")) {
        print();
    }
}

void Tracer::trace() {
    void **buffer = stack.data();
    real_size = backtrace(buffer, max_depth);
    // buffer[0]: address of trace()
    // buffer[1]: address of local hook function (xpu_wait())
    // buffer[2]: address of the caller function of xpu_wait()
    if (real_size < 3) {
        return;
    }
    void *addr = buffer[2];
    Dl_info info;
    if (dladdr(addr, &info)) {
        std::string lib_abs_name = info.dli_fname;
        std::string lib_name = get_file_from_path(lib_abs_name);
        if (Recorder::enable) {
            Recorder *recorder = RecorderSingleton::instance().get_elem();
            if (recorder) {
                recorder->record(lib_name, func_name);
            }
        }
    }
}

void Tracer::print() {
    char **strings = backtrace_symbols(stack.data(), real_size);
    std::cout << "Stack trace:" << std::endl;
    for (int i = 0; i < real_size; ++i) {
        std::cout << strings[i] << std::endl;
    }
    free(strings);
}
