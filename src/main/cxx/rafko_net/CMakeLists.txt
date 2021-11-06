add_library(rafko_net)
target_include_directories(rafko_net PUBLIC ${PROJECT_SOURCE_DIR} models services)
target_link_libraries(rafko_net PUBLIC rafko_protocol)
set(NET_INTERFACE
  models/cost_function_squared_error.h
  models/cost_function_mse.h
  services/synapse_iterator.h
  services/weight_updater_adam.h
  services/weight_updater_amsgrad.h
  services/weight_updater_momentum.h
  services/weight_updater_nesterov.h
)
set(NET_HEADERS
  models/cost_function.h
  models/dense_net_weight_initializer.h
  models/neuron_info.h
  models/spike_function.h #TODO: Generalize and move specialized source to interfaces
  models/transfer_function.h
  models/weight_initializer.h
  services/backpropagation_queue_wrapper.h
  services/function_factory.h
  services/neuron_router.h
  services/partial_solution_builder.h
  services/partial_solution_solver.h
  services/solution_builder.h
  services/solution_solver.h
  services/sparse_net_builder.h
  services/updater_factory.h
  services/weight_experience_space.h
  services/weight_updater.h
)
set(NET_PUBLIC_HEADERS
  ${NET_INTERFACE}
  ${NET_HEADERS}
)
target_sources(rafko_net
  INTERFACE ${NET_INTERFACE}
  PUBLIC ${NET_HEADERS}
  PRIVATE
  models/src/dense_net_weight_initializer.cc
  models/src/neuron_info.cc
  models/src/transfer_function.cc
  models/src/cost_function.cc
  services/src/backpropagation_queue_wrapper.cc
  services/src/neuron_router.cc
  services/src/partial_solution_builder.cc
  services/src/partial_solution_solver.cc
  services/src/solution_builder.cc
  services/src/sparse_net_builder.cc
  services/src/solution_solver.cc
  services/src/weight_experience_space.cc
  services/src/weight_updater.cc
)
install(
  FILES ${NET_PUBLIC_HEADERS}
  DESTINATION include/rafko_net
)
