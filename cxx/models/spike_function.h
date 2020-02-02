#ifndef SPIKE_FUNCTION_H
#define SPIKE_FUNCTION_H

#include <vector>

#include "sparse_net_global.h"

namespace sparse_net_library{

/**
 * @brief      Spike function handling and utilities
 */
class Spike_function{
public:
  /**
   * @brief      Apply the given spike function to a neurons activation data
   *
   * @param[in]  parameter The parameter supplied by a Neuron
   * @param[in]  data      The data to apply it to
   */
  static sdouble32 get_value(sdouble32 parameter, sdouble32 new_data, sdouble32 previous_data){
    return (previous_data * parameter) + (new_data * (1.0-parameter));
  }

  /**
   * @brief      Gets a functions derivative calculated form the given data
   *
   * @param[in]  parameter The parameter supplied by a Neuron
   * @param[in]  data      The data to use
   *
   * @return     The derivative from data.
   */
  static sdouble32 get_derivative(sdouble32 parameter){
    return (1 - parameter);
  }
};
} /* namespace sparse_net_library */
#endif /* SPIKE_FUNCTION_H */
