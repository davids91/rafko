set( MAINFRAME_MODEL_HEADERS
  models/rafko_settings.h
)
add_library(rafko_mainframe)
target_include_directories(rafko_mainframe PUBLIC ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR} models services control)

if(BUILD_MAINFRAME)
  target_link_libraries(rafko_mainframe PUBLIC rafko_protocol)
  target_sources(rafko_mainframe
    INTERFACE
    PUBLIC
    ${MAINFRAME_MODEL_HEADERS}
    services/training_logger.h
    PRIVATE
    control/src/rafko_deep_learning_mainframe.cc
    services/src/training_logger.cc
  )
  install(
    FILES ${MAINFRAME_MODEL_HEADERS}
    DESTINATION include/rafko_mainframe/models
  )
else()
  set_target_properties(rafko_mainframe PROPERTIES LINKER_LANGUAGE CXX)
  target_sources(rafko_mainframe
    INTERFACE
    PUBLIC
    ${MAINFRAME_MODEL_HEADERS}
  )
  install(
    FILES ${MAINFRAME_MODEL_HEADERS}
    DESTINATION include/rafko_mainframe/models
  )
endif()
