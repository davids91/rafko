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
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  option(MAKE_TESTS
    "Enables test build"
    ON
  )
  option(TEST_CONVERGENCE
    "Makes the test builds examines convergence, instead of acceptance tests"
    OFF
  )

endif()

add_subdirectory(rafko_protocol)
add_subdirectory(rafko_utilities)
add_subdirectory(rafko_net)
add_subdirectory(rafko_gym)
#add_subdirectory(rafko_mainframe)

add_library(rafko_lib "")
target_link_libraries(rafko_lib PUBLIC rafko_protocol rafko_utilities rafko_net rafko_gym)
target_include_directories(rafko_lib PUBLIC ${CMAKE_BINARY_DIR}) # For Generated protocol files

add_custom_target(run
  COMMAND ./rafko_test -d yes
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/test
  COMMENT "running tests in ${CMAKE_BINARY_DIR}/test"
)

if(MAKE_TESTS)
  add_subdirectory(test)
endif()
