#include "models/neuron_info.h"

namespace sparse_net_library{

uint32 Neuron_info::get_neuron_estimated_size_bytes(const Neuron& neuron){
  uint32 ret = 0;
  ret = neuron.weight_index_starts_size() * 4/* Bytes */ * 2/* fields (weights and inputs) */;
  ret += neuron.input_index_starts_size() * 2/* Byte */ * 2/* fields( size and starts) */;
  return ret;
}

} /* namespace sparse_net_library */
