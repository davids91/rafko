add_library(rafko_utilities OBJECT)
target_include_directories(rafko_utilities PUBLIC ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}  models services)

if(RAFKO_USES_OPENCL)
  set(RAFKO_GPU_LIBRARY_HEADERS
    models/rafko_gpu_kernel_library.h
  )
else()
  set(RAFKO_GPU_LIBRARY_HEADERS)
endif()

set(UTIL_INTERFACE_MODELS
  models/data_pool.h
  models/data_ringbuffer.h
  models/const_vector_subrange.h
  ${RAFKO_GPU_LIBRARY_HEADERS}
)
set(UTIL_INTERFACE_SERVICES
  services/thread_group.h
  services/rafko_string_utils.h
  services/rafko_math_utils.h
)

target_sources(rafko_utilities
  PRIVATE
  services/src/rafko_string_utils.cc
  services/src/thread_group.cc
)

install(
  FILES ${UTIL_INTERFACE_MODELS}
  DESTINATION include/rafko_utilities/models
)
install(
  FILES ${UTIL_INTERFACE_SERVICES}
  DESTINATION include/rafko_utilities/services
)
