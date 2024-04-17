#include "utils/Timer/Timer.h"
#include "utils/Utils.h"
#include "utils/Log/Log.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <mutex>

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
    void record_time(std::string ph = "B", std::string name = "launch_async", std::string tid = "runtime api", std::string cname = "yellow");
    int64_t get_time();
    void set_size(int64_t size);
    void set_flag();
    void record_duration();
    int64_t get_duration();
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
};

bool Timer::enable = false;

Timer::Timer() : start(high_resolution_clock::now()) {
    pre_time_point = start;
    sec_time_point = pre_time_point;
}

Timer::Timer(int64_t size) : start(high_resolution_clock::now()) {
    pre_time_point = start;
    sec_time_point = pre_time_point;
    times.reserve(size);
}

void Timer::enable_timer() { enable = true; }

void Timer::record_time(std::string ph, std::string name, std::string tid, std::string cname) {
    high_resolution_clock::time_point time_point = std::chrono::high_resolution_clock::now();
    std::unique_lock<std::mutex> lock(mtx);
    times.push_back(std::move(time_point));
    names.push_back(name);
    tids.push_back(tid);
    phs.push_back(ph);
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
    std::unique_lock<std::mutex> lock(mtx);
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

Timer::~Timer() {
    LOG() << "Timer::~Timer()";
    if (times.empty()) {
        LOG() << "No time recorded.";
        return;
    }
    std::ofstream file("/tmp/example.json");
    if (!file.is_open()) {
        std::cerr << "Failed to  open file." << std::endl;
    }

    int64_t time_num = times.size();
    file << "[\n";
    for (int64_t index = 0; index < time_num; index++) {
        Json json;
        if (index != 0) {
            file << ",\n";
        }
        json.add("name", names[index]);
        json.add("ph", phs[index]);
        json.add("tid", tids[index]);
        json.add("pid", "main");
        int64_t duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               times[index] - start)
                               .count();
        json.add("ts", std::to_string((double)duration / 1000));
        file << json.to_string();
    }
    file << "]\n";
    file.close();
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

void record_time(std::string ph , std::string name , std::string tid, std::string cname) {
    if (Timer::enable) {
        TimerSingletone::instance().get_elem()->record_time(ph, name, tid, cname); 
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

} // namespace timer