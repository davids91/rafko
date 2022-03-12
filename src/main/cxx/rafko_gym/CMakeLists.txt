add_library(rafko_gym OBJECT)
target_link_libraries(rafko_gym PUBLIC rafko_protocol rafko_net)
target_include_directories(rafko_gym PUBLIC ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR} models services)
set(GYM_INTERFACE_SERVICES
  services/cost_function_squared_error.h
  services/cost_function_mse.h
  services/cost_function_cross_entropy.h
  services/cost_function_binary_cross_entropy.h
  services/weight_updater_adam.h
  services/weight_updater_amsgrad.h
  services/weight_updater_momentum.h
  services/weight_updater_nesterovs.h
)
set(GYM_HEADER_MODELS
  models/rafko_agent.h
  models/rafko_environment.h
  models/rafko_objective.h
  models/rafko_dataset_wrapper.h
  models/rafko_cost.h
  models/rafko_network_delta_chain_link.h
  models/rafko_network_proxy.h
)
set(GYM_HEADER_SERVICES
  services/cost_function.h
  services/function_factory.h
  services/rafko_net_approximizer.h
  services/updater_factory.h
  services/rafko_weight_updater.h
)
target_sources(rafko_gym
  INTERFACE ${GYM_INTERFACE_SERVICES}
  PUBLIC ${GYM_HEADERS}
  PRIVATE
  models/src/rafko_cost.cc
  models/src/rafko_dataset_wrapper.cc
  models/src/rafko_network_delta_chain_link.cc
  services/src/cost_function.cc
  services/src/rafko_net_approximizer.cc
  services/src/rafko_weight_updater.cc
  services/src/weight_updater_adam.cc
  services/src/weight_updater_amsgrad.cc
)

install(
  FILES ${GYM_INTERFACE_SERVICES} ${GYM_HEADER_SERVICES}
  DESTINATION include/rafko_gym/services
)

install(
  FILES ${GYM_HEADER_MODELS}
  DESTINATION include/rafko_gym/models
)
