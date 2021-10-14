add_library(rafko_mainframe)
target_include_directories(rafko_mainframe PUBLIC ${PROJECT_SOURCE_DIR} models services control)
target_link_libraries(rafko_mainframe PUBLIC rafko_protocol)
target_sources(rafko_mainframe
  INTERFACE
  PUBLIC
  models/service_context.h
  services/deep_learning_server.h
  services/server_slot_approximize_net.h
  services/server_slot_factory.h
  services/server_slot_optimize_net.h
  services/server_slot_run_net.h
  services/server_slot.h
  services/training_logger.h
  PRIVATE
  control/src/rafko_deep_learning_mainframe.cc
  services/src/deep_learning_server.cc
  services/src/server_slot_approximize_net.cc
  services/src/server_slot_run_net.cc
  services/src/server_slot.cc
  services/src/training_logger.cc
)
