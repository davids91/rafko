cmake_minimum_required(VERSION 3.18.4) #Lover Version might be supported...

project(
  rafko
  VERSION 0.6
  DESCRIPTION "A deep learning Framework extended wtih per Neuron memory capabilities with focus on architecture search through training"
  LANGUAGES CXX
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_FLAGS, "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

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

add_subdirectory(rafko_protocol)
add_subdirectory(rafko_utilities)
add_subdirectory(rafko_net)
add_subdirectory(rafko_gym)
if(BUILD_MAINFRAME)
  add_subdirectory(rafko_mainframe)
endif()

add_library(rafko_lib "")
target_link_libraries(rafko_lib PUBLIC rafko_protocol rafko_utilities rafko_net rafko_gym)

if(MAKE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()


set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/lib")
install( TARGETS rafko_lib
  EXPORT rafko-export
  LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
  RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
)
