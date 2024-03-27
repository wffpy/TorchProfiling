#ifndef CONSOLETABLE_CONSOLETABLEENTRY_H
#define CONSOLETABLE_CONSOLETABLEENTRY_H

#include <string>
#include <vector>

namespace console_table {
class ConsoleTableRow {
  public:
    ConsoleTableRow(int width);

    void add_entry(std::string data, int column);

    void edit_entry(std::string data, int column);

    std::vector<std::string> get_entry();

  private:
    std::vector<std::string> row;
};

} // namespace console_table
#endif // CONSOLETABLE_CONSOLETABLEENTRY_H