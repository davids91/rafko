if(MAKE_TESTS)
  add_executable(rafko_test)
  add_test(NAME rafko_test_suite COMMAND rafko_test -d yes)
  target_link_libraries(rafko_test rafko_full)
  target_include_directories(rafko_test PRIVATE ${PROJECT_SOURCE_DIR})

  if(TEST_CONVERGENCE)
    target_sources(rafko_test
      PRIVATE
      src/main_test.cc
      rafko_gym/src/approximizer_test.cc
    )
  else()
    target_sources(rafko_test
      PRIVATE
      src/main_test.cc
      rafko_gym/src/data_aggregate_test.cc
      rafko_gym/src/environment_data_set_test.cc
      rafko_utilities/src/data_pool_test.cc
      rafko_utilities/src/data_ringbuffer_test.cc
      rafko_utilities/src/thread_group_test.cc
      rafko_net/src/backprop_queue_wrapper_test.cc
      rafko_net/src/cost_function_mse_test.cc
      rafko_net/src/neuron_info_test.cc
      rafko_net/src/neuron_router_test.cc
      rafko_net/src/partial_solution_solver_test.cc
      rafko_net/src/solution_builder_test.cc
      rafko_net/src/solution_solver_test.cc
      rafko_net/src/rafko_net_builder_test.cc
      rafko_net/src/synapse_iterator_test.cc
      rafko_net/src/weight_experience_space_test.cc
      rafko_net/src/weight_updater_test.cc
      rafko_mainframe/src/service_context_test.cc
    )
  endif()
endif()
