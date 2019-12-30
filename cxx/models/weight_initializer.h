#ifndef weight_initializer_H
#define weight_initializer_H


#include "gen/sparse_net.pb.h"

#include "sparse_net_global.h"

namespace sparse_net_library {

class Weight_initializer
{
public:
  /**
   * @brief      Constructs the object.
   */
  Weight_initializer() noexcept;

  /**
   * @brief      Calculate a weight which fits the Neuron the most based on the configuration parameters
   *             The basis of the number is the transfer function given in the function argument
   *
   * @param[in]  used_transfer_function  The used transfer function
   *
   * @return     The Calculated weight
   */
  virtual sdouble32 next_weight_for(transfer_functions used_transfer_function) const = 0;

  /**
   * @brief      Calculate a number which fits the Neuron the most based on the configuration parameters
   *
   * @return     The Calculated Memory ratio
   */
  virtual sdouble32 next_memory_ratio() const = 0;

  /**
   * @brief      Calculate a bias which fits the Neuron the most based on the configuration parameters 
   *
   * @return     The Calculated Bias value
   */
  virtual sdouble32 next_bias() const = 0;

  /**
   * @brief      Sets the functions expected parameters
   *
   * @param[in]  expected_input_number             The exponent input number
   * @param[in]  expected_input_maximum_value_     The exponent input maximum
   */
  void set(uint32 expected_input_number, sdouble32 expected_input_maximum_value_);

  /**
   * @brief      Calculate a weight which fits the Neuron the most based on the configuration parameters
   *             The basis of the number is the Transfer_function_info::TRANSFER_FUNCTION_IDENTITY
   *
   * @return     The Calculated Weight value
   */
  sdouble32 next_weight() const;
protected:

  /**
   * @brief      Limits the given weight into the limits used in the Neural Network
   *
   * @param[in]  weight  The weight
   *
   * @return     Limited value
   */
  sdouble32 limit_weight(sdouble32 weight) const;

  /**
   * Number of estimated @Neuron inputs expected 
   */
  uint32 expected_input_number = 0;

  /**
   * Estimated Maximum value of one @Neuron input
   */
  sdouble32 expected_input_maximum_value = 0.0;
};

} /* namespace sparse_net_library */
#endif /* weight_initializer_H */
