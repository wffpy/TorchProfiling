add_subdirectory(ConsoleTable)

add_library(Util STATIC BackTrace.cpp Utils.cpp)
target_link_libraries(Util PRIVATE ConsoleTable)