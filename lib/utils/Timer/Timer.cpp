#include "utils/Timer/Timer.h"
#include "utils/Utils.h"
#include "utils/Log/Log.h"
#include "utils/Lock/FileLock.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <string.h>
#include <sys/stat.h> 

using namespace std::chrono;

namespace timer {
// int64_t get_time_ns () {
//     int64_t cur_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
//             std::chrono::high_resolution_clock::now().time_since_epoch())
//             .count();
//     return cur_time;
// }

// enum class COLORS {
//     GREEN, GREY, YELLOW, 
// };

class AtomicFile {
public:
    AtomicFile() : fd(-1) {}
    AtomicFile(const std::string& file_name);
    AtomicFile(const AtomicFile& file);
    ~AtomicFile();
    AtomicFile& operator=(const AtomicFile&);
    // int64_t write_with_lock(std::string content);
    int64_t write_with_lock(const char* content, int64_t offset = 0);
    int64_t write_non_lock(const char* content);
    bool is_emtpy_brackets();
    void acquire_lock();
    void release_lock();
    void clear_file();
private:
    std::string fp;
    int fd;
};

AtomicFile::AtomicFile(const std::string& file_path) : fp(file_path) {
    fd = open(file_path.c_str(), O_RDWR | O_CREAT, 0666);
    LOG() << "fd: " << fd;
    if (fd < 0) {
        ELOG() << "open file failed" << file_path;
    }

    // struct stat fileStat;
    // if (fstat(fd, &fileStat) == -1) {
    //     close(fd);
    //     ELOG() << "fstat file failed" << file_path;
    // } 

    // if(ftruncate(fd, 0) == -1) {
    //     close(fd);
    //     ELOG() << "ftruncate file failed" << file_path;
    // }
}

AtomicFile::AtomicFile(const AtomicFile& file) {
    this->fd  = open(file.fp.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd < 0) {
        ELOG() << "open file failed" << file.fp;
    }
}

AtomicFile& AtomicFile::operator=(const AtomicFile& file) {
    this->fd = open(file.fp.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd < 0) {
        ELOG() << "open file failed" << file.fp;
    }
    return *this;
}

AtomicFile::~AtomicFile() {
    if (fd > -1) {
        close(fd);
    }
}

int64_t AtomicFile::write_with_lock(const char* content, int64_t offset) {
    CHECK(fd > -1, "not implemented AtomiceFile object with fd: < 0");
    // flock(fd, LOCK_EX);
    lseek(fd, offset, SEEK_END);
    int64_t len = write(fd, content, strlen(content));
    // flock(fd, LOCK_UN);
    return len;
}

int64_t AtomicFile::write_non_lock(const char* content) {
    CHECK(fd > -1, "not implemented AtomiceFile object with fd: < 0");
    lseek(fd, 0L, SEEK_END);
    int64_t len = write(fd, content, strlen(content));
    fsync(fd);
    return len;
}

bool AtomicFile::is_emtpy_brackets() {
    CHECK(fd > -1, "not implemented AtomiceFile object with fd: < 0");
    int64_t offset = lseek(fd, -2, SEEK_END);
    if (offset == -1) {
        ELOG() << "lseek failed, the file may be empty";
    }
    char buffer[2];
    read(fd, buffer, 2);
    if (buffer[0] == '}' && buffer[1] == ']') {
        return false;
    }
    return true;
}

void AtomicFile::acquire_lock() {
    CHECK(fd > -1, "not implemented AtomiceFile object with fd: < 0");
    flock(fd, LOCK_EX);
}

void AtomicFile::release_lock() {
    CHECK(fd > -1, "not implemented AtomiceFile object with fd: < 0");
    flock(fd, LOCK_UN);
}

void AtomicFile::clear_file() {
    if (ftruncate(fd, 0) == -1) {
        close(fd);
        ELOG() << "ftruncate file failed" << fp;
    }
}

int64_t get_rank() {
    const char *rank_str= std::getenv("RANK");
    if (rank_str == nullptr) {
        return 0;
    }
    int64_t rank = std::stoi(rank_str);
    return rank;
}

class Json {
  public:
    Json() {}
    void add(std::string key, std::string value);
    std::string to_string();

  private:
    // name, cname, tid, ts, ph, pid ......
    std::vector<std::pair<std::string, std::string>> key_to_value;
};

std::string Json::to_string() {
    std::string indent = " ";
    std::string result = "";
    result += "{";
    for (auto iter = key_to_value.begin(); iter != key_to_value.end(); ++iter) {
        result += indent;
        if (iter != key_to_value.begin()) {
            result += ", ";
        }

        if (iter->first == "ts") {
            result += "\"" + iter->first  + "\"" + ": "  + iter->second;
        } else {
            result += "\"" + iter->first  + "\"" + ": "  + "\"" + iter->second + "\"";
        }
    }
    result += "}";
    return result;
}

void Json::add(std::string key, std::string value) {
    // key_to_value[key] = value;
    key_to_value.emplace_back(key, value);
}

class Timer {
  public:
    Timer();
    ~Timer();
    Timer(int64_t size);
    high_resolution_clock::time_point record_time(std::string ph = "B", std::string name = "launch_async", std::string tid = "runtime api", std::string cname = "yellow");
    void record_flow_event(high_resolution_clock::time_point time, std::string ph = "s", std::string name = "connect", std::string tid = "runtime api", std::string cname = "yellow");
    void record_time_pair(int64_t ns, std::string name = "launch_async", std::string tid = "runtime api", std::string cname = "yellow");
    int64_t get_time();
    void set_size(int64_t size);
    void set_flag();
    void record_duration();
    int64_t get_duration();
    void write_brackets();
    void set_file_path(const std::string& path);
    void write_json();
    std::mutex mtx;
    static void enable_timer();
    static bool enable;

  private:
    high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point pre_time_point;
    std::chrono::high_resolution_clock::time_point sec_time_point;
    // std::vector<std::pair<std::string, high_resolution_clock::time_point>> times;
    std::vector<high_resolution_clock::time_point> times;
    // std::vector<std::string> pid;
    std::vector<std::string> names;
    std::vector<std::string> phs;
    std::vector<std::string> tids;
    std::vector<std::string> cnames;
    bool flag = false;
    int64_t rank;
    AtomicFile file;
    std::once_flag onceFlag;
};

bool Timer::enable = false;

Timer::Timer() : start(high_resolution_clock::now()), rank(get_rank()) {
    pre_time_point = start;
    sec_time_point = pre_time_point;
}

Timer::Timer(int64_t size) : start(high_resolution_clock::now()), rank(get_rank()) {
    pre_time_point = start;
    sec_time_point = pre_time_point;
    times.reserve(size);
}

void Timer::enable_timer() { enable = true; }

void Timer::set_file_path(const std::string& path) {
    file = AtomicFile(path);
    std::call_once(onceFlag, [&](){
        write_brackets();
    });
}

high_resolution_clock::time_point Timer::record_time(std::string ph, std::string name, std::string tid, std::string cname) {
    high_resolution_clock::time_point time_point = std::chrono::high_resolution_clock::now();
    std::lock_guard<std::mutex> lock(mtx);
    times.push_back(std::move(time_point));
    names.push_back(name);
    tids.push_back(tid);
    phs.push_back(ph);
    cnames.push_back(cname);
    return time_point;
}

void Timer::record_flow_event(high_resolution_clock::time_point time, std::string ph, std::string name, std::string tid, std::string cname) {
    std::lock_guard<std::mutex> lock(mtx);
    times.push_back(std::move(time));
    names.push_back(name);
    tids.push_back(tid);
    phs.push_back(ph);
    cnames.push_back(cname);
}

void Timer::record_time_pair(int64_t ns, std::string name, std::string tid, std::string cname) {
    high_resolution_clock::time_point time_point = std::chrono::high_resolution_clock::now();
    std::chrono::nanoseconds dur(ns);
    auto begin = time_point - dur;
    std::lock_guard<std::mutex> lock(mtx);
    times.push_back(std::move(begin));
    names.push_back(name);
    tids.push_back(tid);
    phs.push_back("B");
    cnames.push_back(cname);

    times.push_back(std::move(time_point));
    names.push_back(name);
    tids.push_back(tid);
    phs.push_back("E");
    cnames.push_back(cname);
}

int64_t Timer::get_time() {
    auto cur = std::chrono::high_resolution_clock::now();
    int64_t duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(cur - start)
            .count();
    return duration;
}

void Timer::set_size(int64_t size) {
    times.reserve(size);
    names.reserve(size);
    phs.reserve(size);
    tids.reserve(size);
}

void Timer::set_flag() { flag = true; }

void Timer::record_duration() {
    std::lock_guard<std::mutex> lock(mtx);
    flag = true;
    pre_time_point = sec_time_point;
    sec_time_point = std::chrono::high_resolution_clock::now();
}

int64_t Timer::get_duration() {
    record_duration();
    if (flag) {
        flag = false;
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   sec_time_point - pre_time_point)
            .count();
    } else {
        return 0;
    }
}

void Timer::write_brackets() {
    if (Timer::enable) {
        lock::do_func_in_one_process([&]() {
            file.clear_file();
            std::string brackets = "[\n]";
            auto len = file.write_non_lock(brackets.c_str());
            if (len < 0) {
                ELOG() << "write_brackets failed";
            }
        });
    }
}

Timer::~Timer() {
    LOG() << "Timer::~Timer(), rank: " << rank;
    if (times.empty()) {
        LOG() << "No time recorded.";
        return;
    }
    int64_t time_num = times.size();
    for (int64_t index = 0; index < time_num; index++) {
        file.acquire_lock();
        Json json;
        std::string j_str = "";
        if (index != 0) {
            j_str = ",\n";
        } else if (!file.is_emtpy_brackets()) {
            j_str = ",\n";
        }

        json.add("name", names[index]);
        json.add("ph", phs[index]);
        json.add("tid", tids[index]);
        json.add("pid", std::to_string(rank));
        int64_t duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               times[index] - start)
                               .count();
        json.add("ts", std::to_string((double)duration / 1000));
        j_str += json.to_string();
        j_str += "]";
        file.write_with_lock(j_str.c_str(), -1);
        file.release_lock();
    }
}

typedef utils::Singleton<Timer> TimerSingletone;

void init_timer(int64_t size) {
    if (Timer::enable) {
        TimerSingletone::instance().get_elem()->set_size(size);
    }
}

int64_t get_time() {
    if (Timer::enable) {
        return TimerSingletone::instance().get_elem()->get_time();
    }
    return 0;
}

high_resolution_clock::time_point record_time(std::string ph , std::string name , std::string tid, std::string cname) {
    high_resolution_clock::time_point time;
    if (Timer::enable) {
        DLOG() <<  "recored_time ph:" << ph << ", name: " << name << ", tid: " << tid << ", cname: " << cname;
        time = TimerSingletone::instance().get_elem()->record_time(ph, name, tid, cname); 
    }
    return time;
}

void record_flow_event(high_resolution_clock::time_point time, std::string ph, std::string name, std::string tid, std::string cname) {
    if (Timer::enable) {
        DLOG() <<  "recored_flow_event ph:" << ph << ", name: " << name << ", tid: " << tid << ", cname: " << cname;
        TimerSingletone::instance().get_elem()->record_flow_event(time, ph, name, tid, cname); 
    }
}

void record_time_pair(int64_t ns, std::string name, std::string tid, std::string cname) {
    if (Timer::enable) {
        DLOG() <<  "recored_pre_time ns:" << ns << ", name: " << name << ", tid: " << tid << ", cname: " << cname;
        TimerSingletone::instance().get_elem()->record_time_pair(ns, name, tid, cname); 
    }
}

void record_duration() {
    if (Timer::enable) {
        TimerSingletone::instance().get_elem()->record_duration();
    }
}

int64_t get_duration() {
    if (Timer::enable) {
        return TimerSingletone::instance().get_elem()->get_duration();
    }
    return 0;
}

void enable_timer() {
    Timer::enable = true;
}

void set_record_path(const std::string& path) {
    if (Timer::enable) {
        TimerSingletone::instance().get_elem()->set_file_path(path);;
    }
}

} // namespace timer