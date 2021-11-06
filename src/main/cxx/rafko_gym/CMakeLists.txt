add_library(rafko_gym)
target_link_libraries(rafko_gym PUBLIC rafko_protocol rafko_net)
target_include_directories(rafko_gym PUBLIC ${PROJECT_SOURCE_DIR} models services)
set(GYM_INTERFACE
  services/environment_data_set.h
)
set(GYM_HEADERS
  models/data_aggregate.h
  services/agent.h
  services/environment.h
  services/sparse_net_approximizer.h
)
set(GYM_PUBLIC_HEADERS
  ${GYM_INTERFACE}
  ${GYM_HEADERS}
)
target_sources(rafko_gym
  INTERFACE ${GYM_INTERFACE}
  PUBLIC ${GYM_HEADERS}
  PRIVATE
  models/src/data_aggregate.cc
  services/src/environment_data_set.cc
  services/src/sparse_net_approximizer.cc
)

install(
  FILES ${GYM_PUBLIC_HEADERS}
  DESTINATION include/rafko_gym
)
