add_library(rafko_net OBJECT)
target_include_directories(rafko_net PUBLIC ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR} models services)
target_link_libraries(rafko_net PUBLIC rafko_mainframe rafko_protocol)
set(NET_INTERFACE_SERVICES
  services/synapse_iterator.hpp
)
set(NET_HEADER_SERVICES
  services/neuron_router.hpp
  services/partial_solution_builder.hpp
  services/partial_solution_solver.hpp
  services/solution_builder.hpp
  services/solution_solver.hpp
  services/rafko_net_builder.hpp
  services/rafko_network_feature.hpp
)
set(NET_HEADER_MODELS
  models/dense_net_weight_initializer.hpp
  models/neuron_info.hpp
  models/spike_function.hpp
  models/transfer_function.hpp
  models/input_function.hpp
  models/weight_initializer.hpp
)
target_sources(rafko_net
  INTERFACE ${NET_INTERFACE_SERVICES}
  PUBLIC ${NET_HEADER_SERVICES} ${NET_HEADER_MODELS}
  PRIVATE
  models/src/neuron_info.cc
  models/src/input_function.cc
  models/src/transfer_function.cc
  models/src/spike_function.cc
  services/src/feature_group_cache.cc
  services/src/neuron_router.cc
  services/src/partial_solution_builder.cc
  services/src/partial_solution_solver.cc
  services/src/solution_builder.cc
  services/src/rafko_net_builder.cc
  services/src/solution_solver.cc
  services/src/rafko_network_feature.cc
)

install(
  FILES ${NET_HEADER_MODELS}
  DESTINATION include/rafko_net/models
)
install(
  FILES ${NET_HEADER_SERVICES} ${NET_INTERFACE_SERVICES}
  DESTINATION include/rafko_net/services
)
