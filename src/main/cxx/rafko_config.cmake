
# Generate global macros for the library
set(LIB_EXPORT_CONSTANTS
"\
/*! This file is part of davids91/Rafko.\
 *\
 *    Rafko is free software: you can redistribute it and/or modify\
 *    it under the terms of the GNU General Public License as published by\
 *    the Free Software Foundation, either version 3 of the License, or\
 *    (at your option) any later version.\
 *\
 *    Rafko is distributed in the hope that it will be useful,\
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of\
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\
 *    GNU General Public License for more details.\
 *\
 *    You should have received a copy of the GNU General Public License\
 *    along with Rafko.  If not, see <https://www.gnu.org/licenses/> or\
 *    <https://github.com/davids91/rafko/blob/master/LICENSE>\
 */\
#ifndef RAFKO_LIB_EXPORT_CONSTANTS_H\n\
#define RAFKO_LIB_EXPORT_CONSTANTS_H\n\
"
)

if(USE_OPENCL)
  set(OPENCL_TARGET 120)
  set(LIB_EXPORT_CONSTANTS
    "${LIB_EXPORT_CONSTANTS}\
    #define CL_HPP_MINIMUM_OPENCL_VERSION ${OPENCL_TARGET}\n\
    #define CL_HPP_TARGET_OPENCL_VERSION ${OPENCL_TARGET}\n\
    #define RAFKO_USES_OPENCL 1\n\
    "
  )
else()
  set(LIB_EXPORT_CONSTANTS
    "${LIB_EXPORT_CONSTANTS}\
    #define RAFKO_USES_OPENCL 0\n\
    "
  )
endif()

if(ASSERTLOGS)
  set(LIB_EXPORT_CONSTANTS
    "${LIB_EXPORT_CONSTANTS}\
    #define RAFKO_USES_ASSERTLOGS 1\n\
    "
  )
else()
  set(LIB_EXPORT_CONSTANTS
    "${LIB_EXPORT_CONSTANTS}\
    #define RAFKO_USES_ASSERTLOGS 0\n\
    "
  )
endif()

set(LIB_EXPORT_CONSTANTS
  "${LIB_EXPORT_CONSTANTS}\
  #endif /*RAFKO_LIB_EXPORT_CONSTANTS_H*/\n\
  "
)

# export constants to header
if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/rafko_config.hpp)
  file(READ ${CMAKE_CURRENT_BINARY_DIR}/rafko_config.hpp WRITTEN_LIB_EXPORT_CONSTANTS)
else()
  set(WRITTEN_LIB_EXPORT_CONSTANTS "")
endif()

if (NOT "${LIB_EXPORT_CONSTANTS}" STREQUAL "${WRITTEN_LIB_EXPORT_CONSTANTS}")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rafko_config.hpp "${LIB_EXPORT_CONSTANTS}")
endif()
