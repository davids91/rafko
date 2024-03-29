add_library(rafko_utilities OBJECT)
target_include_directories(rafko_utilities PUBLIC ${CMAKE_BINARY_DIR} ${PROJECT_SOURCE_DIR}  models services)

if(RAFKO_USES_OPENCL)
  set(RAFKO_GPU_LIBRARY_HEADERS
    models/rafko_gpu_kernel_library.hpp
  )
else()
  set(RAFKO_GPU_LIBRARY_HEADERS)
endif()

set(UTIL_INTERFACE_MODELS
  models/data_pool.hpp
  models/data_ringbuffer.hpp
  models/const_vector_subrange.hpp
  models/subscript_proxy.hpp
  ${RAFKO_GPU_LIBRARY_HEADERS}
)
set(UTIL_INTERFACE_SERVICES
  services/thread_group.hpp
  services/rafko_string_utils.hpp
  services/rafko_math_utils.hpp
)

target_sources(rafko_utilities
  PRIVATE
  services/src/rafko_math_utils.cc
  services/src/rafko_string_utils.cc
  services/src/thread_group.cc
)

# generate convenience header part for current module
set(MODULE_HEADER_LIST)
generate_sum_include("rafko_utilities" "${UTIL_INTERFACE_MODELS};${UTIL_INTERFACE_SERVICES}" MODULE_HEADER_LIST)
list(APPEND MODULE_HEADER_LIST ${SUM_HEADER_FILES})
set(SUM_HEADER_FILES ${MODULE_HEADER_LIST} PARENT_SCOPE)

# install rules
install(
  FILES ${UTIL_INTERFACE_MODELS}
  DESTINATION include/rafko_utilities/models
)
install(
  FILES ${UTIL_INTERFACE_SERVICES}
  DESTINATION include/rafko_utilities/services
)
