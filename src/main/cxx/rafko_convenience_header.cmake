
separate_arguments(SUM_HEADER_FILES)
set(CONV_HEADER_SOURCE
"\
/*! This file is part of davids91/Rafko.\n\
 *\n\
 *    Rafko is free software: you can redistribute it and/or modify\n\
 *    it under the terms of the GNU General Public License as published by\n\
 *    the Free Software Foundation, either version 3 of the License, or\n\
 *    (at your option) any later version.\n\
 *\n\
 *    Rafko is distributed in the hope that it will be useful,\n\
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of\n\
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n\
 *    GNU General Public License for more details.\n\
 *\n\
 *    You should have received a copy of the GNU General Public License\n\
 *    along with Rafko.  If not, see <https://www.gnu.org/licenses/> or\n\
 *    <https://github.com/davids91/rafko/blob/master/LICENSE>\n\
 */\n\
#ifndef RAFKO_FULL_H \n\
#define RAFKO_FULL_H \n\

#include <google/protobuf/arena.h>
")

foreach(HEADER_FILE ${SUM_HEADER_FILES})
  set(CONV_HEADER_SOURCE "${CONV_HEADER_SOURCE}\n#include<${HEADER_FILE}>")
endforeach()

set(CONV_HEADER_SOURCE "${CONV_HEADER_SOURCE}\n\
#endif/*RAFKO_FULL_H*/\
"
)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/rafko.hpp)
  file(READ ${CMAKE_CURRENT_BINARY_DIR}/rafko.hpp WRITTEN_CONV_HEADER_SOURCE)
else()
  set(WRITTEN_LIB_EXPORT_CONSTANTS "")
endif()

if (NOT "${CONV_HEADER_SOURCE}" STREQUAL "${WRITTEN_CONV_HEADER_SOURCE}")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rafko.hpp "${CONV_HEADER_SOURCE}")
endif()
