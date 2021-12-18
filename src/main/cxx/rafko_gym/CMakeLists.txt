add_library(rafko_gym OBJECT)
target_link_libraries(rafko_gym PUBLIC rafko_protocol rafko_net)
target_include_directories(rafko_gym PUBLIC ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR} models services)
set(GYM_INTERFACE_SERVICES
  services/rafko_environment_data_set.h
  services/weight_updater_adam.h
  services/weight_updater_amsgrad.h
  services/weight_updater_momentum.h
  services/weight_updater_nesterovs.h
)
set(GYM_HEADER_MODELS
  models/data_aggregate.h
)
set(GYM_HEADER_SERVICES
  models/data_aggregate.h
  services/rafko_agent.h
  services/rafko_environment.h
  services/rafko_net_approximizer.h
  services/updater_factory.h
  services/rafko_weight_updater.h
  services/backpropagation_queue_wrapper.h
  services/weight_experience_space.h
)
target_sources(rafko_gym
  INTERFACE ${GYM_INTERFACE_SERVICES}
  PUBLIC ${GYM_HEADERS}
  PRIVATE
  models/src/data_aggregate.cc
  services/src/backpropagation_queue_wrapper.cc
  services/src/rafko_environment_data_set.cc
  services/src/rafko_net_approximizer.cc
  services/src/rafko_weight_updater.cc
  services/src/weight_experience_space.cc
)


install(
  FILES ${GYM_INTERFACE_SERVICES} ${GYM_HEADER_SERVICES}
  DESTINATION include/rafko_gym/services
)

install(
  FILES ${GYM_HEADER_MODELS}
  DESTINATION include/rafko_gym/models
)
