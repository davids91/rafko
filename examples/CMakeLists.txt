cmake_minimum_required(VERSION 3.20)
project(rafko_using_networks LANGUAGES CXX)

option(USE_OPENCL "Enables OpenCL specific functionalities" ON)

add_library(rafko STATIC IMPORTED)
set_target_properties(rafko PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/rafko_lib/lib/librafko.a"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/rafko_lib/include"
)

find_package(Protobuf REQUIRED)
set(USED_EXTRA_LIBRARIES ${Protobuf_LIBRARIES})

if(USE_OPENCL)
  find_package(OpenCL REQUIRED)
  list(APPEND USED_EXTRA_LIBRARIES OpenCL::OpenCL)
endif()

# Address sanitizer
# add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
# add_link_options(-fsanitize=address -fno-omit-frame-pointer)
# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

add_executable(create_network create_network.cc)
target_link_libraries(create_network rafko ${USED_EXTRA_LIBRARIES})

add_executable(execute_network execute_network.cc)
target_link_libraries(execute_network rafko ${USED_EXTRA_LIBRARIES})

add_executable(optimize_network optimize_network.cc)
target_link_libraries(optimize_network rafko ${USED_EXTRA_LIBRARIES})
