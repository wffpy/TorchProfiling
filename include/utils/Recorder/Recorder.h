#include <string>
namespace recorder {
void record(std::string s);
void write_to_file();
void set_record_file(std::string file);
void enable_recorder();
void close_recorder();
}  // namespace recoder