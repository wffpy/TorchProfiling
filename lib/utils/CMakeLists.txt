add_subdirectory(ConsoleTable)
add_subdirectory(Log)

add_library(Util STATIC BackTrace.cpp Utils.cpp)
target_link_libraries(Util PRIVATE ConsoleTable)
target_link_libraries(Util PRIVATE Log)