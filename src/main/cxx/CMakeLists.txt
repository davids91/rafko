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
endif()

add_subdirectory(rafko_utilities)
add_subdirectory(rafko_net)
add_subdirectory(rafko_gym)
add_subdirectory(rafko_mainframe)
add_subdirectory(test)

add_library(rafko "")
target_link_libraries(rafko rafko_utilities rafko_net rafko_gym)
install(
  TARGETS rafko_utilities rafko_net rafko_gym rafko_mainframe
  LIBRARY DESTINATION build
  PUBLIC_HEADER DESTINATION build/include
)
