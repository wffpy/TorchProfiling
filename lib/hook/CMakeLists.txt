add_subdirectory(LocalHook)

add_library(hook STATIC CFuncHook.cpp)
target_link_libraries(hook PRIVATE LocalHook)
