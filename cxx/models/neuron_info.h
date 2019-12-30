#ifndef NEURON_INFO_H
#define NEURON_INFO_H

#include "sparse_net_global.h"
#include "gen/sparse_net.pb.h"

namespace sparse_net_library{

class Neuron_info{
public: 
  
  /**
   * @brief      Gets a neurons estimated size in bytes.
   *
   * @param[in]  neuron  The neuron
   *
   * @return     The neuron estimated size in bytes.
   */
  static uint32 get_neuron_estimated_size_bytes(const Neuron& neuron);
  
  /**
   * @brief      Determines whether the specified neuron is valid, but does
   *             not take SparseNet integrity into account (eg.: it doesn't check index validities)
   *
   * @param[in]  neuron  The neuron reference
   */
  static bool is_neuron_valid(const Neuron& neuron);
};

} /* namespace sparse_net_library */

#endif /* NEURON_INFO_H */