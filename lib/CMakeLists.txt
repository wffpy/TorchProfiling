add_library(hook STATIC CFuncHook.cpp BackTrace.cpp RegisterHook.cpp)

    target_include_directories(hook PRIVATE ${Python3_INCLUDE_DIRS})
        target_link_libraries(hook PRIVATE ${Python3_LIBRARIES})
            target_link_libraries(hook PRIVATE GpuHook)

                install(TARGETS hook DESTINATION ${DST_DIRECTORY_TO_INSTALL})

                    add_subdirectory(cuda)
