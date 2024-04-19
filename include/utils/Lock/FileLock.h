#ifndef UTILS_LOCK_FILELOCK_H
#define UTILS_LOCK_FILELOCK_H
#include <string>
#include <functional>

namespace lock {

// do the func in only one process,
// before the func is done, all other process will be blocked
void do_func_in_one_process(std::function<void()>func);

} // namespace lock
#endif //UTILS_LOCK_FILELOCK_H