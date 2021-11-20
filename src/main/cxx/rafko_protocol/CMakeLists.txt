add_library(rafko_protocol OBJECT)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/gen/rafko_protocol)
target_include_directories(rafko_protocol
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gen
)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
set(PROTO_SOURCES
  common.proto
  logger.proto
  solution.proto
  rafko_net.proto
  training.proto
  deep_learning_service.proto
)
target_link_libraries(rafko_protocol
  PUBLIC
    ${Protobuf_LIBRARIES}
    gRPC::grpc
    gRPC::grpc++
)
get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
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

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gen/rafko_protocol
  DESTINATION include/
  FILES_MATCHING PATTERN "*.h"
)
