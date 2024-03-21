#include <execinfo.h>
#include "BackTrace.h"

using namespace trace;
Tracer::Tracer() : max_depth(100), real_size(0) {
    stack.reserve(max_depth);
}

void Tracer::trace() {
    void** buffer = stack.data();
    real_size = backtrace(buffer, max_depth);
}

void Tracer::print() {
    char** strings = backtrace_symbols(stack.data(), real_size);
    std::cout << "Stack trace:" << std::endl;
    for (int i = 0; i < real_size; ++i) {
        std::cout << strings[i] << std::endl;
    }
}

