#ifndef DUMP_H
#define DUMP_H
#include <cstdint>

namespace dump {
void record_tensor(const uint64_t ptr, int64_t size);
}   // namespace dump
#endif