add_library(rafko_mainframe OBJECT)
target_include_directories(rafko_mainframe PUBLIC ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR} models services control)

if(USE_OPENCL)
  find_package(OpenCL REQUIRED)
  target_link_libraries(rafko_mainframe PUBLIC rafko_protocol OpenCL::OpenCL)
  set(HEADERS_MODELS_WITH_OCL
    models/rafko_gpu_strategy.hpp
  )
  set(HEADERS_SERVICES_WITH_OCL
    services/rafko_ocl_factory.hpp
    services/rafko_gpu_context.hpp
    services/rafko_gpu_phase.hpp
  )
  set(SOURCES_WITH_OCL
    models/src/rafko_gpu_strategy.cc
    services/src/rafko_gpu_phase.cc
    services/src/rafko_gpu_context.cc
  )
else()
  target_link_libraries(rafko_mainframe PUBLIC rafko_protocol)
  set(HEADERS_MODELS_WITH_OCL)
  set(HEADERS_SERVICES_WITH_OCL)
  set(SOURCES_WITH_OCL)
endif()

if(ASSERTLOGS)
  add_subdirectory(external/spdlog)
  target_link_libraries(rafko_mainframe PUBLIC spdlog::spdlog)

  add_subdirectory(external/date)
  target_link_libraries(rafko_mainframe PUBLIC date::date)
endif()

set(MAINFRAME_MODELS_HEADERS
  ${HEADERS_MODELS_WITH_OCL}
  models/rafko_settings.hpp
  models/rafko_nbuf_shape.hpp
  models/rafko_autonomous_entity.hpp
)
set(MAINFRAME_SERVICES_HEADERS
  ${HEADERS_SERVICES_WITH_OCL}
  services/rafko_dummies.hpp
  services/rafko_context.hpp
  services/rafko_cpu_context.hpp
  services/rafko_training_logger.hpp
  services/rafko_assertion_logger.hpp
)
set(MAINFRAME_SOURCES
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
    ${MAINFRAME_MODELS_HEADERS}
    ${MAINFRAME_SERVICES_HEADERS}
    ${MAINFRAME_SOURCES}
    PRIVATE
    control/src/rafko_deep_learning_mainframe.cc
  )
else()
  set_target_properties(rafko_mainframe PROPERTIES LINKER_LANGUAGE CXX)
  target_sources(rafko_mainframe
    INTERFACE
    PUBLIC
    ${MAINFRAME_HEADERS}
    ${MAINFRAME_SOURCES}
  )
endif()

# generate convenience header part for current module
set(MODULE_HEADER_LIST)
generate_sum_include("rafko_mainframe" "${MAINFRAME_MODELS_HEADERS};${MAINFRAME_SERVICES_HEADERS}" MODULE_HEADER_LIST)
list(APPEND MODULE_HEADER_LIST ${SUM_HEADER_FILES})
set(SUM_HEADER_FILES ${MODULE_HEADER_LIST} PARENT_SCOPE)

# install rules
install(
  FILES ${MAINFRAME_MODELS_HEADERS}
  DESTINATION include/rafko_mainframe/models
)
install(
  FILES ${MAINFRAME_SERVICES_HEADERS}
  DESTINATION include/rafko_mainframe/services
)
