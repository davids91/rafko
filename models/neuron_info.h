#ifndef NEURON_INFO_H
#define NEURON_INFO_H

#include "sparse_net_global.h"
#include "models/gen/sparse_net.pb.h"


namespace sparse_net_library{

class Neuron_info{
public: 
  static uint32 get_neuron_estimated_size_bytes(const Neuron& neuron);
};

} /* namespace sparse_net_library */

#endif /* NEURON_INFO_H */