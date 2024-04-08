#ifndef LOG_H
#define LOG_H
#include <iostream>
#include <sstream>

namespace log_module {

enum class LogLevel { INFO, DEBUG, WARN, ERROR };

class Logger {
  public:
    Logger(std::string file_name, std::string func_name, int line);
    Logger(std::string file_name, std::string func_name, int line,
           LogLevel level);
    // Logger(LogLevel level);
    ~Logger();
    template <typename T> Logger &operator<<(const T &value) {
        os_stream_ << value;
        return *this;
    }

    void flush();

  private:
    std::string level_str();
    LogLevel level_;
    std::ostringstream os_stream_;
};
} // namespace log_module

#define WLOG()                                                                 \
    log_module::Logger(__FILE__, __FUNCTION__, __LINE__,                       \
                       log_module::LogLevel::INFO)

#define CHECK(cond, msg)                                                       \
    (!(cond)) ? (std::cerr << "Assertion failed: (" << #cond << "), "                                                                        \
                                << "function " << __FUNCTION__ << ", file " << __FILE__ << ", line " << __LINE__ << "." << std::endl                   \
                                << msg << std::endl,                                                                                               \
                        abort(), 0)                                                                                                                      \
                    : 1

#endif // LOG_H