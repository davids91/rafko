add_library(rafko_utilities INTERFACE)
target_include_directories(rafko_utilities INTERFACE ${PROJECT_SOURCE_DIR}  models services)
target_link_libraries(rafko_utilities INTERFACE rafko_protocol)
set(UTIL_INTERFACE_MODELS
  models/data_pool.h
  models/data_ringbuffer.h
)
set(UTIL_INTERFACE_SERVICES
  services/thread_group.h
)
target_sources(rafko_utilities INTERFACE ${UTIL_INTERFACE})

install(
  FILES ${UTIL_INTERFACE_MODELS}
  DESTINATION include/rafko_utilities/models
)
install(
  FILES ${UTIL_INTERFACE_SERVICES}
  DESTINATION include/rafko_utilities/services
)
