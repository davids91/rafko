add_library(rafko_mainframe)
target_include_directories(rafko_mainframe PUBLIC ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR} models services control)

if(USE_OPENCL)
  find_package(OpenCL REQUIRED)
  target_link_libraries(rafko_mainframe PUBLIC rafko_protocol OpenCL::OpenCL)
  set(HEADERS_WITH_OCL
    models/rafko_gpu_strategy_phase.hpp
    services/rafko_ocl_factory.hpp
  )
  set(SOURCES_WITH_OCL
    models/src/rafko_gpu_strategy_phase.cc
    services/src/rafko_gpu_phase.cc
    services/src/rafko_gpu_context.cc
  )
else()
  target_link_libraries(rafko_mainframe PUBLIC rafko_protocol)
  set(HEADERS_WITH_OCL)
  set(SOURCES_WITH_OCL)
endif()

if(ASSERTLOGS)
  add_subdirectory(external/spdlog)
  target_link_libraries(rafko_mainframe PUBLIC spdlog::spdlog)

  add_subdirectory(external/date)
  target_link_libraries(rafko_mainframe PUBLIC date::date)
endif()

set( MAINFRAME_HEADERS
  ${HEADERS_WITH_OCL}
  models/rafko_settings.hpp
  models/rafko_nbuf_shape.hpp
  models/rafko_autonomous_entity.hpp
  services/rafko_dummies.hpp
  services/rafko_context.hpp
  services/rafko_cpu_context.hpp
  services/rafko_training_logger.hpp
  services/rafko_assertion_logger.hpp
)
set( MAINFRAME_SOURCES
  ${SOURCES_WITH_OCL}
  models/src/rafko_settings.cc
  services/src/rafko_cpu_context.cc
  services/src/rafko_training_logger.cc
  services/src/rafko_assertion_logger.cc
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
