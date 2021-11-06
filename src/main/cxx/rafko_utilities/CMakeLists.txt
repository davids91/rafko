add_library(rafko_utilities INTERFACE)
target_include_directories(rafko_utilities INTERFACE ${PROJECT_SOURCE_DIR}  models services)
target_link_libraries(rafko_utilities INTERFACE rafko_protocol)
set(UTIL_INTERFACE
  models/data_pool.h
  models/data_ringbuffer.h
  services/thread_group.h
)
target_sources(rafko_utilities INTERFACE ${UTIL_INTERFACE})

install(
  FILES ${UTIL_INTERFACE}
  DESTINATION include/rafko_utilities
)
