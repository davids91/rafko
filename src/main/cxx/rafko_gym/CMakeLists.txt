add_library(rafko_gym)
target_include_directories(rafko_gym PUBLIC ${PROJECT_SOURCE_DIR} models services)
target_sources(rafko_gym
  INTERFACE
  services/environment_data_set.h
  PUBLIC
  models/data_aggregate.h
  services/agent.h
  services/environment.h
  services/sparse_net_approximizer.h
  PRIVATE
  models/src/data_aggregate.cc
  services/src/environment_data_set.cc
  services/sparse_net_approximizer.h
)
