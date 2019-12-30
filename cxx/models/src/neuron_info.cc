#include "models/neuron_info.h"

namespace sparse_net_library{

uint32 Neuron_info::get_neuron_estimated_size_bytes(const Neuron& neuron){
  uint32 ret = 0;
  ret = neuron.input_weights_size()  * 2/* Byte */ * 2/* fields( interval_size and starts) */;
  ret += neuron.input_indices_size() * 2/* Byte */ * 2/* fields( interval_size and starts) */;
  return ret;
}

bool Neuron_info::is_neuron_valid(const Neuron& neuron)
{
  if(
    (transfer_functions_IsValid(neuron.transfer_function_idx())) /* Transfer Function ID is valid */
    &&(TRANSFER_FUNCTION_UNKNOWN < neuron.transfer_function_idx()) /* Transfer Function ID is known */
    &&(( /* Either the input is consistent */
      (0 < neuron.input_indices_size()) /* There are any inputs */
      &&(0 < neuron.input_weights_size()) /* and weights */
    )||( /* Or there is no input. we won't judge. */
      (0 == neuron.input_indices_size()) && (0 == neuron.input_weights_size()) 
    ))
  ){ /*!Note: Only the first synapse sizes are checked for non-zero size for perfomance purposes. 
      * It is enough to determine if there is any input to the Neuron, because
      * if the first is non-zero then essentially there are more, than 0 inputs.
      */
    
    uint32 number_of_input_indexes = 0;
    for(int i = 0; i<neuron.input_indices_size(); ++i){
      number_of_input_indexes += neuron.input_indices(i).interval_size();
    }

    uint32 number_of_input_weights = 0;
    for(int i = 0; i<neuron.input_weights_size(); ++i){
      number_of_input_weights += neuron.input_weights(i).interval_size();
    }

    /* Check if inputs from synapses match */
    return (number_of_input_indexes == number_of_input_weights);
  } else return false;
}

} /* namespace sparse_net_library */
