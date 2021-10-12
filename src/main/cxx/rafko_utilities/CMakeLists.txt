add_library(rafko_utilities INTERFACE)
target_include_directories(rafko_utilities INTERFACE ${PROJECT_SOURCE_DIR}  models services)
target_sources(rafko_utilities INTERFACE
  models/data_pool.h
  models/data_ringbuffer.h
  services/thread_group.h
)
