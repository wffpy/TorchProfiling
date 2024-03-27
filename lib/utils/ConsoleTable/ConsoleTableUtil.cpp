#include "utils/ConsoleTable/ConsoleTableUtil.h"
using namespace console_table;
namespace console_table {
std::string repeat_string(std::string input, int n) {
    std::ostringstream os;
    for (int i = 0; i < n; i++) {
        os << input;
    }
    return os.str();
}
} // namespace console_table