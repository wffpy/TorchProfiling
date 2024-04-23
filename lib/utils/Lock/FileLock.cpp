#include "utils/Lock/FileLock.h"
#include "utils/Log/Log.h"
#include "utils/Utils.h"
#include <fcntl.h>
#include <functional>
#include <string>
#include <sys/file.h>
#include <unistd.h>

namespace lock {
int acquire_file_descriptor(const std::string &file_path) {
    int fd = open(file_path.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        ELOG() << "Failed to open lock file: " << file_path;
    }
    return fd;
}

// blocked until acquire lock.
bool acquire_file_lock(int fd) {
    if (flock(fd, LOCK_EX) < 0) {
        return false;
    }
    return true;
}

// try to acquire lock, not block if not acquire lock.
bool try_acquire_file_lock(int fd) {
    if (flock(fd, LOCK_EX | LOCK_NB) < 0) {
        close(fd);
        return false;
    }
    return true;
}

// release file lock
void release_file_lock(int fd) { flock(fd, LOCK_UN); }

// use File lock to implement multi process lock.
// 1. all process will be synced
// 2. only one process will execute the given function.
class MultiProcessLock {
  public:
    MultiProcessLock();
    MultiProcessLock(const std::string &file_path1,
                     const std::string &file_path2);
    ~MultiProcessLock();
    void lock(std::function<void()> func);

  private:
    int fd1;
    int fd2;
    bool fd2_lock_flag;
};

MultiProcessLock::MultiProcessLock() : fd2_lock_flag(false) {
    std::string lock_file_1 = "/tmp/lock1";
    std::string lock_file_2 = "/tmp/lock2";
    fd1 = acquire_file_descriptor(lock_file_1);
    fd2 = acquire_file_descriptor(lock_file_2);
}

MultiProcessLock::MultiProcessLock(const std::string &file_path1,
                                   const std::string &file_path2)
    : fd2_lock_flag(false) {
    fd1 = acquire_file_descriptor(file_path1);
    fd2 = acquire_file_descriptor(file_path2);
}

MultiProcessLock::~MultiProcessLock() {
    //  if current process has locked fd2, release flock.
    if (fd2_lock_flag) {
        release_file_lock(fd2);
    }
    // close fd1, fd2 desriptor.
    close(fd1);
    close(fd2);
}

void MultiProcessLock::lock(std::function<void()> func) {
    if (fd1 < 0 || fd2 < 0) {
        ELOG() << "Failed to acquire lock";
    }
    // fd1 can just be locked by one process, and other processes will be
    // blocked by fd1 lock.
    acquire_file_lock(fd1);

    // fd2 will be just be lock by process which 1st acquire lock on fd1 and
    // will not be released.
    if (try_acquire_file_lock(fd2)) {
        fd2_lock_flag = true;
        LOG() << "Lock fd2 success";
        func();
    }
    // release fd1 lock unitl the func is done, so that other processes can
    // continue
    release_file_lock(fd1);
}

typedef utils::Singleton<MultiProcessLock> MPLSingleton;

void do_func_in_one_process(std::function<void()> func) {
    MPLSingleton::instance().get_elem()->lock(func);
}

} // namespace lock