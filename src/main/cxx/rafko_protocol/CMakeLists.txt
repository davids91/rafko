find_package(Protobuf REQUIRED CONFIG)
if(NOT Protobuf_LIBRARIES)
  set(Protobuf_LIBRARIES "protobuf::libprotobuf")
endif()

add_library(rafko_protocol OBJECT)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/gen/rafko_protocol)
target_include_directories(rafko_protocol
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gen
)
if(BUILD_MAINFRAME)
  include(FetchContent)
  FetchContent_Declare(
    gRPC
    GIT_REPOSITORY https://github.com/grpc/grpc
    GIT_TAG        v1.43.2
  )
  set(FETCHCONTENT_QUIET OFF)
  FetchContent_MakeAvailable(gRPC)
  set(DEEP_LEARNING_SERVICE_PROTOS
    deep_learning_service.proto
  )
  set(GRPC_LIBRARIES
    grpc
    grpc++
  )
else()
  set(DEEP_LEARNING_SERVICE_PROTOS)
  set(GRPC_LIBRARIES)
endif()

set(PROTO_SOURCES
  logger.proto
  solution.proto
  rafko_net.proto
  training.proto
  ${DEEP_LEARNING_SERVICE_PROTOS}
)

target_link_libraries(rafko_protocol PUBLIC ${Protobuf_LIBRARIES} ${GRPC_LIBRARIES})
protobuf_generate(
  TARGET
    rafko_protocol
  LANGUAGE
    cpp
  GENERATE_EXTENSIONS
    .pb.cc .pb.h
  PROTOC_OUT_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/gen/rafko_protocol
  PROTOS
    ${PROTO_SOURCES}
)
if(BUILD_MAINFRAME)
  protobuf_generate(
    TARGET
      rafko_protocol
    LANGUAGE
      grpc
    GENERATE_EXTENSIONS
      .grpc.pb.cc .grpc.pb.h
    PROTOC_OUT_DIR
      ${CMAKE_CURRENT_SOURCE_DIR}/gen/rafko_protocol
    PROTOS
      deep_learning_service.proto
    PLUGIN
      "protoc-gen-grpc=${grpc_cpp_plugin_location}"
  )
endif()

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gen/rafko_protocol
  DESTINATION include/
  FILES_MATCHING PATTERN "*.h"
)
