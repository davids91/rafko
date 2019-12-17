
#include "services/solution_builder.h"

namespace sparse_net_library{

Solution_builder& Solution_builder::used_devices_number(uint32 number){

  return *this;
}

Solution_builder& Solution_builder::device_available_byte_size(uint32 device_index, uint32 size){

  return *this;
}

Solution* Solution_builder::build( const SparseNet* net ){
  throw NOT_IMPLEMENTED_EXCEPTION;
}

} /* namespace sparse_net_library */
