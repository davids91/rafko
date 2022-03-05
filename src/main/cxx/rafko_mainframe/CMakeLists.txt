add_library(rafko_mainframe)
target_include_directories(rafko_mainframe PUBLIC ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR} models services control)

if(USE_OPENCL)
  find_package(OpenCL REQUIRED)
  target_link_libraries(rafko_mainframe PUBLIC rafko_protocol OpenCL::OpenCL)
  set(CONTEXT_HEARDERS_OCL
    models/rafko_gpu_strategy_phase.h
    services/src/rafko_gpu_phase.cc
  )
  set(CONTEXT_SOURCES_OCL
    models/src/rafko_gpu_strategy_phase.cc
    services/src/rafko_gpu_context.cc
  )
else()
  target_link_libraries(rafko_mainframe PUBLIC rafko_protocol)
  set(CONTEXT_HEARDERS_OCL)
  set(CONTEXT_SOURCES_OCL)
endif()

set( MAINFRAME_HEADERS
  ${CONTEXT_HEARDERS_OCL}
  models/rafko_settings.h
  models/rafko_nbuf_shape.h
  services/rafko_dummies.h
  services/rafko_context.h
  services/rafko_cpu_context.h
  services/rafko_training_logger.h
  services/rafko_assertion_logger.h
)
set( MAINFRAME_SOURCES
  ${CONTEXT_SOURCES_OCL}
  models/src/rafko_settings.cc
  services/src/rafko_cpu_context.cc
  services/src/training_logger.cc
)
if(BUILD_MAINFRAME)
  target_sources(rafko_mainframe
    INTERFACE
    PUBLIC
    ${MAINFRAME_HEADERS}
    ${MAINFRAME_SOURCES}
    PRIVATE
    control/src/rafko_deep_learning_mainframe.cc
  )
  install(
    FILES ${MAINFRAME_HEADERS}
    DESTINATION include/rafko_mainframe/models
  )
else()
  set_target_properties(rafko_mainframe PROPERTIES LINKER_LANGUAGE CXX)
  target_sources(rafko_mainframe
    INTERFACE
    PUBLIC
    ${MAINFRAME_HEADERS}
    ${MAINFRAME_SOURCES}
  )
  install(
    FILES ${MAINFRAME_HEADERS}
    DESTINATION include/rafko_mainframe/models
  )
endif()
