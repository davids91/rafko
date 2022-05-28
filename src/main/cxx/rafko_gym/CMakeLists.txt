add_library(rafko_gym OBJECT)
target_link_libraries(rafko_gym PUBLIC rafko_protocol rafko_net)
target_include_directories(rafko_gym PUBLIC ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR} models services)

if(USE_OPENCL)
  find_package(OpenCL REQUIRED)
  target_link_libraries(rafko_gym PUBLIC rafko_protocol OpenCL::OpenCL)
  set(GYM_HEADERS_WITH_OCL
    services/rafko_autodiff_gpu_strategy.h
    services/rafko_autodiff_gpu_optimizer.h
  )
  set(GYM_SOURCES_WITH_OCL
    services/src/rafko_autodiff_gpu_strategy.cc
  )
else()
  target_link_libraries(rafko_mainframe PUBLIC rafko_protocol)
  set(GYM_HEADERS_WITH_OCL)
  set(GYM_SOURCES_WITH_OCL)
endif()

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
)
set(GYM_HEADER_SERVICES
  services/cost_function.h
  services/function_factory.h
  services/rafko_numeric_optimizer.h
  services/updater_factory.h
  services/rafko_weight_adapter.h
  services/rafko_weight_updater.h
  services/rafko_autodiff_optimizer.h
  services/rafko_backpropagation_operation.h
  services/rafko_backrpop_solution_feature_operation.h
  services/rafko_backprop_weight_reg_operation.h
  services/rafko_backprop_network_input_operation.h
  services/rafko_backprop_neuron_bias_operation.h
  services/rafko_backprop_neuron_input_operation.h
  services/rafko_backprop_transfer_fn_operation.h
  services/rafko_backprop_spike_fn_operation.h
  services/rafko_backprop_objective_operation.h
)
target_sources(rafko_gym
  INTERFACE ${GYM_INTERFACE_SERVICES}
  PUBLIC ${GYM_HEADERS}
  PRIVATE
  ${GYM_SOURCES_WITH_OCL}
  models/src/rafko_cost.cc
  models/src/rafko_dataset_wrapper.cc
  services/src/cost_function.cc
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

install(
  FILES ${GYM_INTERFACE_SERVICES} ${GYM_HEADER_SERVICES}
  DESTINATION include/rafko_gym/services
)

install(
  FILES ${GYM_HEADER_MODELS}
  DESTINATION include/rafko_gym/models
)
