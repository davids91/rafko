#include "services/sparse_net_solver.h"
#include "models/transfer_function_info.h"

namespace sparse_net_library {

std::unique_ptr<sdouble32> SparseNetSolver::solve(SparseNet const * net){
  throw NOT_IMPLEMENTED_EXCEPTION;
}

std::unique_ptr<sdouble32> SparseNetSolver::solve(std::vector<sdouble32> output, const SparseNet *net){

  throw NOT_IMPLEMENTED_EXCEPTION;
}

std::unique_ptr<sdouble32> SparseNetSolver::calculate_spikes(SparseNet const * net){
  throw NOT_IMPLEMENTED_EXCEPTION;
}

std::unique_ptr<sdouble32> SparseNetSolver::calculate_spikes(std::vector<sdouble32> output, const SparseNet *net){
  throw NOT_IMPLEMENTED_EXCEPTION;
}

std::vector<sdouble32> SparseNetSolver::solveDetail(std::unique_ptr<Solution_detail> detail){
	if(nullptr != detail){
    sdouble32 new_neuron_data = 0;
    /* Go through the neurons */
    uint32 neuron_inputs_start_index = 0;
    for(uint8 neuron_iter = 0; neuron_iter < detail->internal_neuron_number; neuron_iter++){
      /* sum the inputs of the neuron together */
      for(uint32 input_iter = 0; input_iter < detail->input_sizes[neuron_iter]; input_iter++){
        new_neuron_data += ( /* Weight of the input * data of the input */
          detail->weights[neuron_inputs_start_index + input_iter]
          *detail->data[detail->inside_indexes[neuron_inputs_start_index + input_iter]]
        );
      }
      neuron_inputs_start_index += detail->input_sizes[neuron_iter];

      /* Add bias */
      new_neuron_data += detail->biases[neuron_iter];

      /* Apply transfer function */
      Transfer_function_info::apply_to_data(detail->transfer_functions[neuron_iter], new_neuron_data);

      /* Apply memory ratio */
      detail->data[neuron_iter] = (
        (detail->data[neuron_iter] * detail->memory_ratios[neuron_iter])
        + (new_neuron_data * (1.0-detail->memory_ratios[neuron_iter]))
      );
    }

    return std::vector<sdouble32>(
      detail->data.data() + detail->input_data_size,
      detail->data.data() + detail->input_data_size + detail->internal_neuron_number
    );
	}else{
		throw NULL_DETAIL_EXCEPTION;
	}
}

bool SparseNetSolver::is_detail_valid(Solution_detail* detail){
	if(
		(nullptr != detail)
		&&(0 < detail->internal_neuron_number)
		&&(nullptr != &detail->input_sizes[detail->internal_neuron_number - 1])
	){
		uint32 weight_array_size = 0;
		uint32 weight_array_it = 0;
		for(uint16 neuron_iterator = 0; neuron_iterator < detail->internal_neuron_number; neuron_iterator++){
			weight_array_size += detail->input_sizes[neuron_iterator]; /* Calculate how many inputs the neuron shall have altogether */
		}

		if(nullptr != &detail->inside_indexes[weight_array_size]){
			/* Check if the inputs for every Neuron are before its index.
			 * This will ensure that there are no unresolved dependencies are present at any Neuron
			 **/
			for(uint16 neuron_iterator = 0; neuron_iterator < detail->internal_neuron_number; neuron_iterator++){
				for(uint32 neuron_input_iterator = 0; neuron_input_iterator < detail->input_sizes[neuron_iterator]; neuron_input_iterator++){
					if(detail->inside_indexes[weight_array_it] > neuron_iterator){ /* Self-recurrence is simulated by adding the current data of a nauron as an input into the solution detail */
						return false;
					}
					weight_array_it++;
				}
			}
		}else return false;

		return(
			(0 < detail->input_data_size)
			&&(nullptr != &detail->data[detail->internal_neuron_number + detail->input_data_size - 1])

			&&(nullptr != &detail->actual_index[detail->internal_neuron_number - 1])
			&&(nullptr != &detail->transfer_functions[detail->internal_neuron_number - 1])
			&&(nullptr != &detail->memory_ratios[detail->internal_neuron_number - 1])
			&&(nullptr != &detail->biases[detail->internal_neuron_number - 1])

			&&(nullptr != &detail->weights[weight_array_size])
		);
	}else return false;
}


} /* namespace sparse_net_library */
