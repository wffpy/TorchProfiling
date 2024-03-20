add_library(hook STATIC KernelHook.cpp)

target_include_directories(hook PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(hook PRIVATE ${Python3_LIBRARIES})

install(TARGETS hook DESTINATION ${DST_DIRECTORY_TO_INSTALL})