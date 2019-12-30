#include "services/partial_solution_solver.h"

#include <algorithm>
#include <cmath>

#include "models/transfer_function_info.h"

namespace sparse_net_library {

void Partial_solution_solver::reset(void){
  neuron_output = vector<sdouble32>(detail.get().internal_neuron_number());
  input_iterator = Synapse_iterator(detail.get().input_data());
  internal_iterator = Synapse_iterator(detail.get().inside_indices());
  uint32 input_size = 0;
  input_iterator.iterate([&](unsigned int synapse_size){
    input_size += synapse_size;
  },[&](int synapse_index){});
  collected_input_data = vector<sdouble32>(input_size);
}

void Partial_solution_solver::collect_input_data(vector<sdouble32>& input_data, vector<sdouble32> neuron_data){
  uint32 input_index = 0;
  input_iterator.iterate([&](int synapse_index){
    if(Synapse_iterator::is_index_input(synapse_index)){ /* If @Partial_solution input is from the network input */
      collected_input_data[input_index] = input_data[Synapse_iterator::input_index_from_synapse_index(synapse_index)];
    }else if(neuron_data.size() > static_cast<std::size_t>(synapse_index)){  /* If @Partial_solution input is from the previous row */
      collected_input_data[input_index] = neuron_data[synapse_index];
    }
    ++input_index;
  });
}

vector<sdouble32> Partial_solution_solver::solve(){
  sdouble32 new_neuron_data = 0;
  sdouble32 new_neuron_input;
  uint32 index_synapse_iterator_start = 0; /* Which is the first synapse belonging to the neuron under @neuron_iterator */
  uint32 weight_synapse_index = 0; /* Which synapse is being processed inside the Neuron */
  uint32 weight_index = 0;

  for(uint16 neuron_iterator = 0; neuron_iterator < detail.get().internal_neuron_number(); ++neuron_iterator){
    new_neuron_data = 0;
    internal_iterator.iterate_unsafe([&](int synapse_index){
      if(Synapse_iterator::is_index_input(synapse_index)){ /* Neuron gets its input from the partialsolution input */
        new_neuron_input = collected_input_data[Synapse_iterator::input_index_from_synapse_index(synapse_index)];
      }else{ /* Neuron gets its input internaly */
        new_neuron_input = neuron_output[synapse_index];
      }

      new_neuron_data += new_neuron_input * /* Data of the input * weight of the input * */
      detail.get().weight_table(
        detail.get().weight_indices(weight_synapse_index).starts() + weight_index
      );

      ++weight_index; /* Step the Weight index forwards */
      if(weight_index >= detail.get().weight_indices(weight_synapse_index).interval_size()){
        weight_index = 0; /* In case the next weight would ascend above the current patition, go to next one */
        ++weight_synapse_index;

        /*!Note: It is possible, in case of an incorrect configuration that the indexes and synapses
         * don't match. It is possible to increase the @weight_synapse_index above @detail.get().weight_synapse_number(neuron_iterator)
         * but that isn't chekced here, mainly for performance reasons.
         **/
      }
    },index_synapse_iterator_start, detail.get().index_synapse_number(neuron_iterator));
    index_synapse_iterator_start += detail.get().index_synapse_number(neuron_iterator);

    /* Add bias */
    new_neuron_data += detail.get().weight_table(detail.get().bias_index(neuron_iterator));

    /* Apply transfer function */
    Transfer_function_info::apply_to_data(detail.get().neuron_transfer_functions(neuron_iterator), new_neuron_data);
    /* Apply memory ratio */
    neuron_output[neuron_iterator] = (
      (neuron_output[neuron_iterator] * detail.get().weight_table(detail.get().memory_ratio_index(neuron_iterator)))
      + (new_neuron_data * (1.0-detail.get().weight_table(detail.get().memory_ratio_index(neuron_iterator))))
    );
  } /* Go through the neurons */
  return neuron_output;
}

uint32 Partial_solution_solver::get_input_size(void) const{
  return collected_input_data.size();
}

bool Partial_solution_solver::is_valid(void){
  if(
    (0u < detail.get().internal_neuron_number())
    &&(static_cast<int>(detail.get().internal_neuron_number()) == detail.get().index_synapse_number_size())
    &&(static_cast<int>(detail.get().internal_neuron_number()) == detail.get().weight_synapse_number_size())
    &&(static_cast<int>(detail.get().internal_neuron_number()) == detail.get().actual_index_size())
    &&(static_cast<int>(detail.get().internal_neuron_number()) == detail.get().neuron_transfer_functions_size())
    &&(static_cast<int>(detail.get().internal_neuron_number()) == detail.get().memory_ratio_index_size())
    &&(static_cast<int>(detail.get().internal_neuron_number()) == detail.get().bias_index_size())
  ){
    int weight_synapse_number = 0;
    int index_synapse_number = 0;

    for(uint16 neuron_iterator = 0u; neuron_iterator < detail.get().internal_neuron_number(); ++neuron_iterator){
      weight_synapse_number += detail.get().weight_synapse_number(neuron_iterator); /* Calculate how many inputs the neuron shall have altogether */
      index_synapse_number += detail.get().index_synapse_number(neuron_iterator); /* Calculate how many inputs the neuron shall have altogether */
    }

    if((0 < index_synapse_number)&&(0 < weight_synapse_number)){
      /* Check if the inputs for every Neuron are before its index.
       * This will ensure that there are no unresolved dependencies are present at any Neuron
       **/
      uint32 index_synapse_iterator_start = 0;
      uint32 count_of_input_indexes = 0;
      uint32 weight_synapse_iterator_start = 0;
      uint32 count_of_input_weights = 0;
      for(uint32 neuron_iterator = 0; neuron_iterator < detail.get().internal_neuron_number(); neuron_iterator++){
        count_of_input_indexes = 0;
        count_of_input_weights = 0;
        for(uint32 internal_iterator = 0; internal_iterator < detail.get().index_synapse_number(neuron_iterator); ++internal_iterator){
          count_of_input_indexes += detail.get().inside_indices(index_synapse_iterator_start + internal_iterator).interval_size();
          if( /* If a synapse input in a Neuron points after the neurons index */
            (detail.get().inside_indices(index_synapse_iterator_start + internal_iterator).starts()
             + detail.get().inside_indices(index_synapse_iterator_start + internal_iterator).interval_size() ) >= neuron_iterator
          ){ /* Self-recurrence is simulated by adding the current data of a neuron as an input into the solution detail */
            return false;
          }

          /* Check if the number of weights match the number of input indexes for every Neuron */
          for(uint32 internal_iterator = 0; internal_iterator < detail.get().weight_synapse_number(neuron_iterator); ++internal_iterator){
            count_of_input_weights +=  detail.get().weight_indices(weight_synapse_iterator_start + internal_iterator).interval_size();
          }
          weight_synapse_iterator_start += detail.get().weight_synapse_number(neuron_iterator);
          index_synapse_iterator_start += detail.get().index_synapse_number(neuron_iterator);

          if(count_of_input_indexes == count_of_input_weights){
            return false;
          }
        }
      }
    }else return false;

    return(
      (index_synapse_number == detail.get().inside_indices_size())
      &&(weight_synapse_number == detail.get().weight_indices_size())
    );
  }else return false;
}

} /* namespace sparse_net_library */
