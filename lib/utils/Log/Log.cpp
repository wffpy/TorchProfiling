#include "utils/Log/Log.h"
#include <iostream>

using namespace log_module;

Logger::Logger(std::string file_name, std::string func_name, int line)
    : level_(LogLevel::INFO), os_stream_(std::ostringstream()) {
    os_stream_ << "[" << file_name << "]";
    os_stream_ << "[" << func_name << "]";
    os_stream_ << "[" << line << "] ";
    os_stream_ << level_str() << ": ";
}

Logger::Logger(std::string file_name, std::string func_name, int line,
               LogLevel level)
    : level_(level), os_stream_(std::ostringstream()) {
    os_stream_ << "[" << file_name << "]";
    os_stream_ << "[" << func_name << "]";
    os_stream_ << "[" << line << "] ";
    os_stream_ << level_str() << ": ";
}

void Logger::flush() {
    std::cout << os_stream_.str() << std::endl;
    os_stream_.clear();
}

Logger::~Logger() {
    if (!os_stream_.str().empty()) {
        std::cout << os_stream_.str() << std::endl;
    }
    if (level_ == LogLevel::ERROR) {
        std::exit(-1);
    }
}

std::string Logger::level_str() {
    std::string type_str = "INFO";
    switch (level_) {
    case LogLevel::INFO:
        type_str = "[INFO]";
        break;
    case LogLevel::DEBUG:
        type_str = "[DEBUG]";
        break;
    case LogLevel::WARN:
        type_str = "[WARNING]";
        break;
    case LogLevel::ERROR:
        type_str = "[ERROR]";
        break;
    default:
        type_str = "[INFO]";
        break;
    }
    return type_str;
}