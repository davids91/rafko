add_library(rafko_net OBJECT)
target_include_directories(rafko_net PUBLIC ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR} models services)
target_link_libraries(rafko_net PUBLIC rafko_protocol)
set(NET_INTERFACE_SERVICES
  services/synapse_iterator.h
)
set(NET_INTERFACE_MODELS
  models/cost_function_squared_error.h
  models/cost_function_mse.h
)
set(NET_HEADER_SERVICES
  services/function_factory.h
  services/neuron_router.h
  services/partial_solution_builder.h
  services/partial_solution_solver.h
  services/solution_builder.h
  services/solution_solver.h
  services/rafko_net_builder.h
)
set(NET_HEADER_MODELS
  models/cost_function.h
  models/dense_net_weight_initializer.h
  models/neuron_info.h
  models/spike_function.h #TODO: Generalize and move specialized source to interfaces
  models/transfer_function.h
  models/weight_initializer.h
)
target_sources(rafko_net
  INTERFACE ${NET_INTERFACE_SERVICES}
  PUBLIC ${NET_HEADER_SERVICES} ${NET_HEADER_MODELS}
  PRIVATE
  models/src/dense_net_weight_initializer.cc
  models/src/neuron_info.cc
  models/src/transfer_function.cc
  models/src/cost_function.cc
  services/src/feature_group_cache.cc
  services/src/neuron_router.cc
  services/src/partial_solution_builder.cc
  services/src/partial_solution_solver.cc
  services/src/solution_builder.cc
  services/src/rafko_net_builder.cc
  services/src/solution_solver.cc
)

install(
  FILES ${NET_HEADER_MODELS} ${NET_INTERFACE_MODELS}
  DESTINATION include/rafko_net/models
)
install(
  FILES ${NET_HEADER_SERVICES} ${NET_INTERFACE_SERVICES}
  DESTINATION include/rafko_net/services
)
