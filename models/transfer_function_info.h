#ifndef Transfer_function_info_H
#define Transfer_function_info_H

#include <vector>

#include "sparse_net_global.h"
#include "models/gen/sparse_net.pb.h"

namespace sparse_net_library {

using std::vector;

class Transfer_function_info
{
public:
  static sdouble32 epsilon; /* very small positive value almost greater, than 0.0 */
  static sdouble32 alpha; 
  static sdouble32 lambda;

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
  static void apply_to_data(transfer_functions function, sdouble32& data);
};

} /* namespace sparse_net_library */
#endif // Transfer_function_info_H
