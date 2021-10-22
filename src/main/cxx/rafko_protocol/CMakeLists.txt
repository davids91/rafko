add_library(rafko_protocol)
target_include_directories(rafko_protocol
  PUBLIC
  .
  ${CMAKE_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Protobuf_INCLUDE_DIRS}
)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
set(PROTO_SOURCES
  common.proto
  logger.proto
  solution.proto
  sparse_net.proto
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
  PROTOS
    ${PROTO_SOURCES}
)
protobuf_generate(
  TARGET
    rafko_protocol
  LANGUAGE
    grpc
  GENERATE_EXTENSIONS
    .grpc.pb.h
    .grpc.pb.cc
  PROTOS
    deep_learning_service.proto
  PLUGIN
    "protoc-gen-grpc=${grpc_cpp_plugin_location}"
)

#protobuf_generate_cpp(
#  LIB_SOURCES LIB_HEADERS
#  ${PROTO_SOURCES}
#  PLUGIN protoc-gen-grpc=${GRPC_CPP_PLUGIN_PATH}
#)
#target_sources(rafko_protocol
#  PRIVATE
#  ${LIB_HEADERS}
#  ${LIB_SOURCES}
#)
