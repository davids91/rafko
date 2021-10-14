add_library(rafko_protocol)
target_include_directories(rafko_protocol
  PUBLIC
  .
  ${CMAKE_BINARY_DIR}
  ${Protobuf_INCLUDE_DIRS}
)
find_package(Protobuf REQUIRED)
set(PROTO_SOURCES
  common.proto
  logger.proto
  solution.proto
  sparse_net.proto
  training.proto
  deep_learning_service.proto
)
protobuf_generate_cpp(LIB_SOURCES LIB_HEADERS ${PROTO_SOURCES} )
target_link_libraries(rafko_protocol ${Protobuf_LIBRARIES})
target_sources(rafko_protocol
  PRIVATE
  ${LIB_HEADERS}
  ${LIB_SOURCES}
)
