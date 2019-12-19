
#include <algorithm>

#include "services/partial_solution_solver.h"
#include "models/transfer_function_info.h"

namespace sparse_net_library {

vector<sdouble32> Partial_solution_solver::solve(const Partial_solution* detail, const vector<sdouble32>* input_data){
  if(
    (nullptr != detail)
    &&(
      (detail->input_data_size() == input_data->size())
      ||(detail->input_data_size() + detail->internal_neuron_number() == input_data->size())
    )
  ){
    vector<sdouble32> data;
    data.reserve(detail->input_data_size() + detail->internal_neuron_number());
    data.insert(data.begin(),input_data->begin(),input_data->end());

    if(detail->input_data_size() == input_data->size()){ /* Fill in values with 0 in case it's not given */
      for(uint32 i = 0; i < detail->internal_neuron_number(); ++i){
        data.push_back(0.0);
      }
    }

    /* Go through the neurons */
    sdouble32 new_neuron_data = 0;
    uint32 index_synapse_iterator_start = 0; /* Which is the first synapse belonging to the neuron under @neuron_iterator */
    uint32 weight_synapse_iterator_index = 0; /* Which synapse is being processed inside the Neuron */
    uint32 weight_index = 0;
    for(
      uint8 neuron_iterator = 0;
      neuron_iterator < detail->internal_neuron_number();
      ++neuron_iterator
    ){
      new_neuron_data = data[detail->input_data_size() + neuron_iterator]; /* Start with the Neurons previous data */
      for(
        uint32 index_synapse_iterator = 0;
        index_synapse_iterator < detail->index_synapse_number(neuron_iterator);
        ++index_synapse_iterator
      ){
        for(
          uint32 input_iterator = 0;
          input_iterator < detail->inside_index_sizes(index_synapse_iterator_start + index_synapse_iterator);
          ++input_iterator
        ){
          new_neuron_data += ( /* Weight of the input * data of the input */
            detail->weight_table(detail->weight_index_starts(weight_synapse_iterator_index) + weight_index)
            * data[detail->inside_index_starts(index_synapse_iterator_start + index_synapse_iterator) + input_iterator]
          );

          weight_index++; /* Step the Weight index forwards */
          if(weight_index >= detail->weight_index_sizes(weight_synapse_iterator_index)){
            weight_index = 0; /* In case the next weight would ascend above the current patition, go to next one */
            weight_synapse_iterator_index++;

            /*!Note: It is possible, in case of an incorrect configuration that the indexes and synapses
             * don't match. It is possible to increase the @weight_synapse_iterator_index above @detail->weight_synapse_number(neuron_iterator)
             * but that isn't chekced here, mainly for performance reasons.
             **/
          }
        } /* For Every input inside a synapse */
      } /* For every synapse inside a Neuron */
      index_synapse_iterator_start += detail->index_synapse_number(neuron_iterator);

      /* Add bias */
      new_neuron_data += detail->weight_table(detail->bias_index(neuron_iterator));

      /* Apply transfer function */
      Transfer_function_info::apply_to_data(detail->neuron_transfer_functions(neuron_iterator), new_neuron_data);

      /* Apply memory ratio */
      data[detail->input_data_size() + neuron_iterator] = (
        (data[detail->input_data_size() + neuron_iterator] * detail->weight_table(detail->memory_ratio_index(neuron_iterator)))
        + (new_neuron_data * (1.0-detail->weight_table(detail->memory_ratio_index(neuron_iterator))))
      );
    }

    return vector<sdouble32>(
      data.data() + detail->input_data_size() + 1,
      data.data() + detail->input_data_size() + detail->internal_neuron_number()
    );
  }else if(nullptr != detail){
    throw "Provided Partial Solution is inconsistent! ";
  }else{
    throw "Provided Partial Solution is a null pointer!";
  }
}

bool Partial_solution_solver::is_valid(const Partial_solution* detail){
  if(
    (0 < detail->input_data_size())
    &&(0u < detail->internal_neuron_number())
    &&(static_cast<int>(detail->internal_neuron_number()) == detail->index_synapse_number_size())
    &&(static_cast<int>(detail->internal_neuron_number()) == detail->weight_synapse_number_size())
    &&(static_cast<int>(detail->internal_neuron_number()) == detail->actual_index_size())
    &&(static_cast<int>(detail->internal_neuron_number()) == detail->neuron_transfer_functions_size())
    &&(static_cast<int>(detail->internal_neuron_number()) == detail->memory_ratio_index_size())
    &&(static_cast<int>(detail->internal_neuron_number()) == detail->bias_index_size())
  ){
    int weight_synapse_number = 0;
    int index_synapse_number = 0;

    for(uint16 neuron_iterator = 0u; neuron_iterator < detail->internal_neuron_number(); neuron_iterator++){
      weight_synapse_number += detail->weight_synapse_number(neuron_iterator); /* Calculate how many inputs the neuron shall have altogether */
      index_synapse_number += detail->index_synapse_number(neuron_iterator); /* Calculate how many inputs the neuron shall have altogether */
    }

    if((0 < index_synapse_number)&&(0 < weight_synapse_number)){
      /* Check if the inputs for every Neuron are before its index.
       * This will ensure that there are no unresolved dependencies are present at any Neuron
       **/
      uint32 index_synapse_iterator_start = 0;
      uint32 count_of_input_indexes = 0;
      uint32 weight_synapse_iterator_start = 0;
      uint32 count_of_input_weights = 0;
      for(uint32 neuron_iterator = 0; neuron_iterator < detail->internal_neuron_number(); neuron_iterator++){
        count_of_input_indexes = 0;
        count_of_input_weights = 0;
        for(uint32 synapse_iterator = 0; synapse_iterator < detail->index_synapse_number(neuron_iterator); ++synapse_iterator){
          count_of_input_indexes += detail->inside_index_sizes(index_synapse_iterator_start + synapse_iterator);
          if( /* If a synapse input in a Neuron points after the neurons index */
            (detail->inside_index_starts(index_synapse_iterator_start + synapse_iterator)
             + detail->inside_index_sizes(index_synapse_iterator_start + synapse_iterator) ) >= neuron_iterator
          ){ /* Self-recurrence is simulated by adding the current data of a neuron as an input into the solution detail */
            return false;
          }

          /* Check if the number of weights match the number of input indexes for every Neuron */
          for(uint32 synapse_iterator = 0; synapse_iterator < detail->weight_synapse_number(neuron_iterator); ++synapse_iterator){
            count_of_input_weights +=  detail->weight_index_sizes(weight_synapse_iterator_start + synapse_iterator);
          }
          weight_synapse_iterator_start += detail->weight_synapse_number(neuron_iterator);
          index_synapse_iterator_start += detail->index_synapse_number(neuron_iterator);

          if(count_of_input_indexes == count_of_input_weights){
            return false;
          }
        }
      }
    }else return false;

    return(
      (index_synapse_number == detail->inside_index_starts_size())
      &&(weight_synapse_number == detail->weight_index_starts_size())
    );
  }else return false;
}

} /* namespace sparse_net_library */
