
#include "services/partial_solution_solver.h"
#include "models/transfer_function_info.h"

namespace sparse_net_library {

vector<sdouble32> Partial_solution_solver::solve(const Partial_solution* detail, const vector<sdouble32>* input_data){
  if(
    (nullptr != detail)
    &&(detail->input_data_size() == input_data->size())
  ){
    vector<sdouble32> data;
    sdouble32 new_neuron_data = 0;
    data.reserve(detail->input_data_size() + detail->internal_neuron_number());
    data.insert(data.begin(),input_data->begin(),input_data->end());

    /* Go through the neurons */
    uint32 neuron_inputs_start_index = 0;
    for(uint8 neuron_iter = 0; neuron_iter < detail->internal_neuron_number(); neuron_iter++){
      /* sum the inputs of the neuron together */
      new_neuron_data = 0;
      for(uint32 input_iter = 0; input_iter < detail->input_sizes(neuron_iter); input_iter++){
        new_neuron_data += ( /* Weight of the input * data of the input */
          detail->weight_table(detail->weight_indexes(neuron_inputs_start_index + input_iter))
          * data[detail->inside_indexes(neuron_inputs_start_index + input_iter)]
        );
      }
      neuron_inputs_start_index += detail->input_sizes(neuron_iter);

      /* Add bias */
      new_neuron_data += detail->biases(neuron_iter);

      /* Apply transfer function */
      Transfer_function_info::apply_to_data(detail->neuron_transfer_functions(neuron_iter), new_neuron_data);

      /* Apply memory ratio */
      data[detail->input_data_size() + neuron_iter] = (
        (data[detail->input_data_size() + neuron_iter] * detail->memory_ratios(neuron_iter))
        + (new_neuron_data * (1.0-detail->memory_ratios(neuron_iter)))
      );
    }

    return vector<sdouble32>(
      data.data() + detail->input_data_size(),
      data.data() + detail->input_data_size() + detail->internal_neuron_number()
    );
  }else if(nullptr != detail){
    throw INVALID_USAGE_EXCEPTION;
  }else{
    throw NULL_DETAIL_EXCEPTION;
  }
}

bool Partial_solution_solver::is_valid(const Partial_solution* detail){
  if(
    (0 < detail->input_data_size())
    &&(0u < detail->internal_neuron_number())
    &&(static_cast<int>(detail->internal_neuron_number()) == detail->input_sizes_size())
    &&(static_cast<int>(detail->internal_neuron_number()) == detail->actual_index_size())
    &&(static_cast<int>(detail->internal_neuron_number()) == detail->neuron_transfer_functions_size())
    &&(static_cast<int>(detail->internal_neuron_number()) == detail->memory_ratios_size())
    &&(static_cast<int>(detail->internal_neuron_number()) == detail->biases_size())
  ){
    int weight_array_size = 0;
    uint32 weight_array_it = 0u;
    for(uint16 neuron_iterator = 0u; neuron_iterator < detail->internal_neuron_number(); neuron_iterator++){
      weight_array_size += detail->input_sizes(neuron_iterator); /* Calculate how many inputs the neuron shall have altogether */
    }

    if(0 < weight_array_size){
      /* Check if the inputs for every Neuron are before its index.
       * This will ensure that there are no unresolved dependencies are present at any Neuron
       **/
      for(uint16 neuron_iterator = 0; neuron_iterator < detail->internal_neuron_number(); neuron_iterator++){
        for(uint32 neuron_input_iterator = 0; neuron_input_iterator < detail->input_sizes(neuron_iterator); neuron_input_iterator++){
          if(detail->inside_indexes(weight_array_it) > neuron_iterator){ /* Self-recurrence is simulated by adding the current data of a neuron as an input into the solution detail */
            return false;
          }
          weight_array_it++;
        }
      }
    }else return false;

    return(
      (weight_array_size == detail->inside_indexes_size())
      &&(weight_array_size == detail->weight_indexes_size())
    );
  }else return false;
}

} /* namespace sparse_net_library */
