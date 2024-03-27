#ifndef CONSOLETABLE_CONSOLETABLE_H
#define CONSOLETABLE_CONSOLETABLE_H

#include "utils/ConsoleTable/ConsoleTableRow.h"
#include "utils/ConsoleTable/ConsoleTableUtil.h"
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace console_table {
enum TableStyle {
    BASIC,
    LINED,
    DOUBLE_LINE,
};

enum HorizontalSeperator { SEPERATOR_TOP, SEPERATOR_MIDDLE, SEPERATOR_BOTTOM };

// reference:
// https://codereview.stackexchange.com/questions/191032/console-based-table-structure
class ConsoleTable {
  public:
    typedef std::shared_ptr<ConsoleTableRow> RowPtr;

    ConsoleTable(TableStyle style);

    void set_padding(unsigned int width);

    void add_column(std::string name);

    void add_row(RowPtr item);

    bool remove_row(int index);

    bool edit_row(std::string data, int row, int col);

    void print_table();

  private:
    unsigned int padding = 1;

    std::vector<std::string> columns;
    std::vector<RowPtr> entries;

    // Table Style variables
    std::string style_line_horizontal;
    std::string style_line_vertical;
    std::string style_line_cross;
    std::string style_t_intersect_right;
    std::string style_t_intersect_left;
    std::string style_t_intersect_top;
    std::string style_t_intersect_bottom;
    std::string style_edge_topleft;
    std::string style_edge_topright;
    std::string style_edge_buttomleft;
    std::string style_edge_buttomright;

    void printHorizontalSeperator(const std::vector<int> &maxWidths,
                                  HorizontalSeperator seperator) const;

    void setTableStyle(TableStyle style);
};

} // namespace console_table

#endif // CONSOLETABLE_CONSOLETABLE_H
