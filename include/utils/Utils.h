#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

namespace utils {

template<typename TYPE>
static void set_static_value(...) {}

template<typename TYPE, typename = decltype(TYPE::enable)>
static void set_static_value(int) {
    TYPE::enable = false;
}

template <typename TYPE> class Singleton {
public:
    static Singleton& instance();
    Singleton() : elem(new TYPE()) {}
    ~Singleton() { 
        set_static_value<TYPE>(0);
        delete elem; 
    }
    TYPE *get_elem();

private:
    TYPE* elem;
};

template <typename TYPE>
Singleton<TYPE>&  Singleton<TYPE>::instance() {
    static Singleton inst = Singleton();
    return inst;
}

template <typename TYPE> TYPE *Singleton<TYPE>::get_elem() { return elem; }

// template <typename TYPE>
// void print_table(TYPE t) {
//     std::cout << "not supported type" << std::endl;
//     exit(-1);
// }

template <typename TYPE>
void print_table(const std::string table_name, const std::vector<std::string> col_name, const std::vector<std::vector<TYPE>>& data) {
    // print table name
    std::cout <<  std::setw(20) << table_name << std::endl; 
    for (auto& col : col_name) {
        std::cout << std::setw(10) << col;
    }
    std::cout << std::endl;

    for (auto& row : data) {
        for (auto& elem : row) {
            std::cout << std::setw(10) << elem;
        }
        std::cout << std::endl;
    }
}

template <typename TYPE>
void print_line(std::initializer_list<TYPE> data) {
    for (auto& elem : data) {
        std::cout << std::setw(20) << std::setfill(' ') << elem;
    }
    std::cout << std::endl;
}

void print_table_name(const std::string t_name);
} // namespace utils
#endif
