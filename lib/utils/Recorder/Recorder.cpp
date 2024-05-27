#include "utils/Recorder/Recorder.h"
#include "utils/Utils.h"
#include "utils/Log/Log.h"
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <list>
#include <mutex>
#include <cstring>
#include <libgen.h>
#include <sys/stat.h>

namespace recorder {
template <typename T>
class Recorder {
public:
    Recorder(std::string file = "");
    ~Recorder();
    void record(T content);
    void write_to_file();
    void clear();
    void set_file_path(std::string file);
    static void enable_recorder();
    static bool enable;
private:
    int fd;
    std::string file_name;
    std::list<T> records;
    std::mutex mtx;
};

template <typename T>
bool Recorder<T>::enable = false;

template <typename T>
Recorder<T>::Recorder(std::string file) : file_name(file), fd(-1) {}

template <typename T>
Recorder<T>::~Recorder() {
    if (Recorder<T>::enable) {
        if (records.size() > 0) {
            write_to_file();
            clear();
        }
    }
    if (fd != -1) {
        close(fd);
    }
}

template <typename T>
void Recorder<T>::enable_recorder() {
    enable = true;
}

template <typename T>
void Recorder<T>::record(T content) {
    std::lock_guard<std::mutex> lock(mtx);
    records.push_back(content);
}

template <typename T>
void Recorder<T>::write_to_file() {
    std::lock_guard<std::mutex> lock(mtx);
    LOG() << "write to file: " << file_name << "\n";
    std::ofstream out(file_name);
    for (auto record : records) {
        out << record << "\n";
    }
}

template <typename T>
void Recorder<T>::clear() {
    std::lock_guard<std::mutex> lock(mtx);
    records.clear();
}

template <typename T>
void Recorder<T>::set_file_path(std::string file) {
    std::lock_guard<std::mutex> lock(mtx);
    file_name = file;
    char* path = dirname((char*)file.c_str());
    if (access(path, F_OK) == -1) {
        mkdir(path, 0777);
    }
}

typedef Recorder<std::string> StrRecorder;
typedef utils::Singleton<StrRecorder> StrRecorderSingleton;

void record(std::string s) {
    if (StrRecorder::enable)
        StrRecorderSingleton::instance().get_elem()->record(s);;
}

void write_to_file() {
    if (StrRecorder::enable) {
        StrRecorderSingleton::instance().get_elem()->write_to_file();;
        StrRecorderSingleton::instance().get_elem()->clear();;
    }
}

void set_record_file(std::string file) {
    if (StrRecorder::enable) {
        StrRecorderSingleton::instance().get_elem()->set_file_path(file);
    }
}

void enable_recorder() {
    StrRecorderSingleton::instance().get_elem()->enable_recorder();
}

}  // namespace recorder