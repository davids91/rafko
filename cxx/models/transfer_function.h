#ifndef TRANSFER_FUNCTION_H
#define TRANSFER_FUNCTION_H

#include <vector>

#include "sparse_net_global.h"
#include "gen/sparse_net.pb.h"

namespace sparse_net_library{

using std::vector;

/**
 * @brief      Transfer function handling and utilities
 */
class Transfer_function{
public:
  /**
   * @brief      Gives a random Transfer Function
   *
   * @return     A random Transfer Function
   */
  static transfer_functions next();

  /**
   * @brief      Provides a random Transfer function out of the ones in the argument
   *
   * @param[in]  range  The range of transfer functions to be given back
   *
   * @return     A random Transfer function according to the given range
   */
  static transfer_functions next(vector<transfer_functions> range);

  /**
   * @brief      Provides the average range of the given Transfer functions output
   *
   * @param[in]  function  The transfer function in question
   *
   * @return     The average output range.
   */
  static sdouble32 get_average_output_range(transfer_functions function);

  /**
   * @brief      Apply the given transfer function to the given data
   *
   * @param[in]  function  The function to apply
   * @param[in]  data      The data to apply it to
   *
   * @return     The result of data.
   */
  static sdouble32 get_value(transfer_functions function, sdouble32 data);

  /**
   * @brief      Gets a functions derivative calculated form the given data
   *
   * @param[in]  function  The function to use
   * @param[in]  data      The data to use
   *
   * @return     The derivative from data.
   */
  static sdouble32 apply_derivative(transfer_functions function, sdouble32 data);
};

} /* namespace sparse_net_library */
#endif /* TRANSFER_FUNCTION_H */
