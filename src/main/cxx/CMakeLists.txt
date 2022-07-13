cmake_minimum_required(VERSION 3.18.4) #Lower Version might be supported...

project(
  rafko
  VERSION 0.0.0 #Have not started versioning yet
  DESCRIPTION "A deep learning Framework extended wtih per Neuron memory capabilities with focus on architecture search through training"
  LANGUAGES CXX
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_CXX_VISIBILITY_INLINES_HIDDEN YES)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(ENABLE_EXPORTS ON)

  option(USE_OPENCL
    "Enables OpenCL specific functionalities"
    ON
  )

  option(BUILD_MAINFRAME
    "Enables build for the included grpc based deep learning service"
    OFF
  )

  option(MAKE_TESTS
    "Enables test build"
    ON
  )

  option(ASSERTLOGS
    "Enables verbose logging for development assertions; Warning: May hike runtime"
    OFF
  )

  option(STATIC_ANALYZERS
    "Enables CPPCheck diagnostics for the project"
    OFF
  )

  option(SANITIZE
    "Enables Gcc address sanitizer to check internal behavior"
    OFF
  )

  add_library(rafko_full STATIC)
  add_compile_options(-Wall -Wextra -Wpedantic)
  set_target_properties(rafko_full PROPERTIES LINKER_LANGUAGE CXX)

  if(USE_OPENCL)
    set(OPENCL_TARGET 120)
    add_compile_definitions(CL_HPP_MINIMUM_OPENCL_VERSION=120)
    add_compile_definitions(CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_TARGET})
    add_compile_definitions(RAFKO_USES_OPENCL=1)
  endif()

  if(ASSERTLOGS)
    add_compile_definitions(RAFKO_USES_ASSERTLOGS=1)
  else()
    add_compile_definitions(RAFKO_USES_ASSERTLOGS=0)
  endif()

  if(SANITIZE)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address -fno-omit-frame-pointer)
  endif()

  if(STATIC_ANALYZERS)
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "include-what-you-use;--verbose=1;")
    set(CMAKE_CXX_CPPCHECK "cppcheck")
  endif()
endif()
include(GenerateExportHeader)
generate_export_header(rafko_full EXPORT_FILE_NAME rafko_export.hpp)

target_compile_features(rafko_full
  PRIVATE
  cxx_constexpr
  cxx_auto_type
  cxx_generic_lambdas
  cxx_lambdas
  cxx_inline_namespaces
  cxx_nullptr
  cxx_raw_string_literals #used in defining OpenCL Kernel source codes
  cxx_range_for
  cxx_rvalue_references
  cxx_sizeof_member
  cxx_static_assert
  cxx_variadic_macros #used in logged assertions
  cxx_variadic_templates # used in log assertions
)

add_subdirectory(rafko_protocol)
add_subdirectory(rafko_utilities)
add_subdirectory(rafko_net)
add_subdirectory(rafko_gym)
add_subdirectory(rafko_mainframe)
target_link_libraries(rafko_full
  PUBLIC rafko_protocol rafko_utilities rafko_net rafko_gym rafko_mainframe
)

if(MAKE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

install(
  FILES rafko_global.hpp ${CMAKE_CURRENT_BINARY_DIR}/rafko_export.hpp
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
