#include "utils/Log/Log.h"
#include <iostream>
#include <string>

using namespace log_module;

#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"

int64_t get_log_level() {
    const char *print_level = std::getenv("LOGLEVEL");
    if (print_level == nullptr) {
        return 0;
    }
    int64_t level = std::stoi(print_level);
    return level;
}

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

void Logger::gen_prefix() {
    switch (level_) {
        case LogLevel::INFO:
            os_stream_ << GREEN;
            break;
        case LogLevel::WARN:
            os_stream_ << YELLOW;
            break;
        case LogLevel::DEBUG:
            os_stream_ << BLUE;
            break;
        case LogLevel::ERROR:
            os_stream_ << RED;
            break;
        default:
            exit(-1);
            break;
    }
}

void Logger::flush() {
    std::cout << os_stream_.str() << std::endl;
    os_stream_.clear();
}

Logger::~Logger() {
    static int64_t log_level = get_log_level();
    if (!os_stream_.str().empty()) {
        switch (level_) {
            case LogLevel::INFO:
                flush();
                break;
            case LogLevel::WARN: {
                if (log_level >= 1) {
                    flush();
                }
                break;
            }
            case LogLevel::DEBUG:{
                if (log_level >= 2) {
                    flush();
                }
                break;
            }
            case LogLevel::ERROR: {
                flush();
                exit(-1);
                break;
            }
            default: {
                std::cout << "not supported log level" << std::endl;
                exit(-1);
            }
        }
    }
}

std::string Logger::level_str() {
    std::string type_str = "INFO";
    switch (level_) {
    case LogLevel::INFO:
        type_str = GREEN;
        type_str += "[INFO]";
        break;
    case LogLevel::DEBUG:
        type_str = BLUE;
        type_str += "[DEBUG]";
        break;
    case LogLevel::WARN:
        type_str = YELLOW;
        type_str += "[WARNING]";
        break;
    case LogLevel::ERROR:
        type_str = RED;
        type_str += "[ERROR]";
        break;
    default:
        type_str = "[INFO]";
        break;
    }
    type_str += RESET;
    return type_str;
}