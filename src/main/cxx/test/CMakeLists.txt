if(MAKE_TESTS)
  add_executable(rafko_test)
  add_subdirectory(external/Catch2)
  set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/test/external/Catch2/extras" ${CMAKE_MODULE_PATH})

  target_include_directories(rafko_test PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(rafko_test PRIVATE rafko Catch2::Catch2)

  include(CTest)
  include(Catch)
  catch_discover_tests(rafko_test)

  if(USE_OPENCL)
    set(GPU_TEST_SOURCES
      rafko_mainframe/src/rafko_gpu_context_test.cc
      rafko_mainframe/src/rafko_gpu_phase_test.cc
    )
  else()
    set(GPU_TEST_SOURCES)
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set( DEBUG_ONLY_TESTS
      rafko_mainframe/src/rafko_assertion_logger_test.cc
    )
  else()
    set(DEBUG_ONLY_TESTS)
  endif()

  target_sources(rafko_test
    PRIVATE
    src/main_test.cc
    rafko_utilities/src/data_pool_test.cc
    rafko_utilities/src/data_ringbuffer_test.cc
    rafko_utilities/src/thread_group_test.cc
    rafko_utilities/src/const_vector_subrange_test.cc
    rafko_utilities/src/subscript_proxy_test.cc
    rafko_utilities/src/rafko_ndarray_index_test.cc
    rafko_net/src/synapse_iterator_test.cc
    rafko_net/src/neuron_router_test.cc
    rafko_net/src/rafko_net_builder_test.cc
    rafko_net/src/neuron_info_test.cc
    rafko_net/src/transfer_functions_test.cc
    rafko_net/src/partial_solution_solver_test.cc
    rafko_net/src/solution_builder_test.cc
    rafko_net/src/solution_solver_test.cc
    rafko_net/src/softmax_function_test.cc
    rafko_net/src/rafko_regularization_tests.cc
    rafko_net/src/rafko_weight_updater_test.cc
    rafko_net/src/rafko_weight_adapter_test.cc
    rafko_gym/src/rafko_dataset_cost_test.cc
    rafko_gym/src/rafko_dataset_wrapper_test.cc
    rafko_gym/src/cost_function_mse_test.cc
    rafko_gym/src/cost_function_cross_entropy_test.cc
    rafko_gym/src/cost_function_binary_cross_entropy_test.cc
    rafko_mainframe/src/rafko_settings_test.cc
    rafko_mainframe/src/rafko_cpu_context_test.cc
    rafko_gym/src/rafko_numeric_optimizer_test.cc
    rafko_gym/src/rafko_autodiff_optimizer_test.cc
    ${GPU_TEST_SOURCES}
    ${DEBUG_ONLY_TESTS}
  )
endif()
