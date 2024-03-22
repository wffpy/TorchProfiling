#ifndef UTILS_H
#define UTILS_H

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
        std::cout << "Singleton::~Singleton" << std::endl; 
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

}  // namespace utils

#endif
