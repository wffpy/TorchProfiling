# # add_library(INIT SHARED init.cpp)
# # add_subdirectory(pybind11)
# pybind11_add_module(INIT init.cpp)

# target_include_directories(INIT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# target_link_libraries(INIT PRIVATE hook)

# target_include_directories(INIT PUBLIC ${Python3_INCLUDE_DIRS})

# target_link_libraries(INIT PRIVATE ${Python3_LIBRARIES})

# # install(TARGETS INIT DESTINATION ${DST_DIRECTORY_TO_INSTALL})

# set_target_properties(INIT
#     PROPERTIES
#     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/module_logging
# )

add_subdirectory(module_logging)