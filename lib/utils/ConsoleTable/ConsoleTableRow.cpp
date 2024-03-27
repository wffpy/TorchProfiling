#include "utils/ConsoleTable/ConsoleTableRow.h"
using namespace console_table;

ConsoleTableRow::ConsoleTableRow(int width) { this->row.resize(width); }

void ConsoleTableRow::add_entry(std::string data, int column) {
    row[column] = data;
}

std::vector<std::string> ConsoleTableRow::get_entry() { return this->row; }

void ConsoleTableRow::edit_entry(std::string data, int column) {
    this->row[column] = data;
}