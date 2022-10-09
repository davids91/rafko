add_library(rafko_gym OBJECT)
target_include_directories(rafko_gym PUBLIC ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR} models services)

if(USE_OPENCL)
  find_package(OpenCL REQUIRED)
  target_link_libraries(rafko_gym PUBLIC rafko_protocol rafko_net OpenCL::OpenCL)
  set(GYM_HEADERS_SERVICES_WITH_OCL
    services/rafko_autodiff_gpu_strategy.hpp
    services/rafko_autodiff_gpu_optimizer.hpp
  )
  set(GYM_SOURCES_WITH_OCL
    services/src/rafko_autodiff_gpu_strategy.cc
    services/src/rafko_autodiff_gpu_optimizer.cc
  )
else()
  target_link_libraries(rafko_gym PUBLIC rafko_protocol rafko_net)
  set(GYM_HEADERS_SERVICES_WITH_OCL)
  set(GYM_SOURCES_WITH_OCL)
endif()

set(GYM_INTERFACE_SERVICES
  services/cost_function_squared_error.hpp
  services/cost_function_mse.hpp
  services/cost_function_cross_entropy.hpp
  services/cost_function_binary_cross_entropy.hpp
  services/weight_updater_adam.hpp
  services/weight_updater_amsgrad.hpp
  services/weight_updater_momentum.hpp
  services/weight_updater_nesterovs.hpp
)
set(GYM_HEADER_MODELS
  models/rafko_agent.hpp
  models/rafko_environment.hpp
  models/rafko_objective.hpp
  models/rafko_dataset_wrapper.hpp
  models/rafko_cost.hpp
  models/rafko_network_delta_chain_link.h
  models/rafko_network_proxy.h
)
set(GYM_HEADER_SERVICES
  ${GYM_HEADERS_SERVICES_WITH_OCL}
  services/cost_function.hpp
  services/function_factory.hpp
  services/rafko_numeric_optimizer.hpp
  services/updater_factory.hpp
  services/rafko_weight_adapter.hpp
  services/rafko_weight_updater.hpp
  services/rafko_autodiff_optimizer.hpp
  services/rafko_backpropagation_operation.hpp
  services/rafko_backprop_solution_feature_operation.hpp
  services/rafko_backprop_weight_reg_operation.hpp
  services/rafko_backprop_network_input_operation.hpp
  services/rafko_backprop_neuron_bias_operation.hpp
  services/rafko_backprop_neuron_input_operation.hpp
  services/rafko_backprop_transfer_fn_operation.hpp
  services/rafko_backprop_spike_fn_operation.hpp
  services/rafko_backprop_objective_operation.hpp
)
target_sources(rafko_gym
  INTERFACE ${GYM_INTERFACE_SERVICES}
  PUBLIC ${GYM_HEADER_SERVICES} ${GYM_HEADER_MODELS}
  PRIVATE
  ${GYM_SOURCES_WITH_OCL}
  models/src/rafko_cost.cc
  models/src/rafko_environment.cc
  models/src/rafko_dataset_wrapper.cc
  models/src/rafko_network_delta_chain_link.cc
  services/src/cost_function.cc
  services/src/rafko_backpropagation_operation.cc
  services/src/rafko_backprop_neuron_input_operation.cc
  services/src/rafko_backprop_neuron_bias_operation.cc
  services/src/rafko_backprop_weight_reg_operation.cc
  services/src/rafko_backprop_spike_fn_operation.cc
  services/src/rafko_backprop_solution_feature_operation.cc
  services/src/rafko_autodiff_optimizer.cc
  services/src/rafko_numeric_optimizer.cc
  services/src/rafko_weight_adapter.cc
  services/src/rafko_weight_updater.cc
  services/src/weight_updater_adam.cc
  services/src/weight_updater_amsgrad.cc
)

# generate convenience header part for current module
set(MODULE_HEADER_LIST)
generate_sum_include("rafko_gym" "${GYM_INTERFACE_SERVICES};${GYM_HEADER_SERVICES};${GYM_HEADER_MODELS}" MODULE_HEADER_LIST)
list(APPEND MODULE_HEADER_LIST ${SUM_HEADER_FILES})
set(SUM_HEADER_FILES ${MODULE_HEADER_LIST} PARENT_SCOPE)

# install rules
install(
  FILES ${GYM_INTERFACE_SERVICES} ${GYM_HEADER_SERVICES}
  DESTINATION include/rafko_gym/services
)

install(
  FILES ${GYM_HEADER_MODELS}
  DESTINATION include/rafko_gym/models
)
