cmake_minimum_required(VERSION 3.18.4) #Lower Version might be supported...

project(
  rafko
  VERSION 0.0.0 #Have not started versioning yet
  DESCRIPTION "A deep learning Framework extended wtih per Neuron memory capabilities with focus on architecture search through training"
  LANGUAGES CXX
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_EXTENSIONS OFF)
  add_compile_options(-Wall -Wextra -Wpedantic)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_CXX_VISIBILITY_INLINES_HIDDEN YES)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(ENABLE_EXPORTS ON)

  option(MAKE_TESTS
    "Enables test build"
    ON
  )
  option(TEST_CONVERGENCE
    "Makes the test builds examines convergence, instead of acceptance tests"
    OFF
  )
  option(BUILD_MAINFRAME
    "Enables build for the included grpc based deep learning service"
    OFF
  )

endif()
include(GenerateExportHeader)
find_package(Protobuf REQUIRED)
add_library(rafko_full STATIC)
generate_export_header(rafko_full EXPORT_FILE_NAME rafko_export.h)

add_subdirectory(rafko_protocol)
add_subdirectory(rafko_utilities)
add_subdirectory(rafko_net)
add_subdirectory(rafko_gym)
add_subdirectory(rafko_mainframe)

set_target_properties(rafko_full PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(rafko_full PUBLIC rafko_protocol rafko_utilities rafko_net rafko_gym rafko_mainframe ${Protobuf_LIBRARIES})

if(MAKE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

install(
  FILES rafko_global.h ${CMAKE_CURRENT_BINARY_DIR}/rafko_export.h
  DESTINATION include/
)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/lib")
install( TARGETS rafko_full
  EXPORT rafko_export
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
  RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
)
