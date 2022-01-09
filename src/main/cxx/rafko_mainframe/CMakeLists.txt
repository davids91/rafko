add_library(rafko_mainframe)
target_include_directories(rafko_mainframe PUBLIC ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR} models services control)
target_link_libraries(rafko_mainframe PUBLIC rafko_protocol)

set( MAINFRAME_HEADERS
  models/rafko_settings.h
  services/rafko_dummies.h
  services/rafko_context.h
  services/rafko_cpu_context.h
  services/training_logger.h
)
set( MAINFRAME_SOURCES
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
