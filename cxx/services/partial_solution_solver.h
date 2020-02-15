#ifndef Partial_solution_H
#define Partial_solution_H

#include <vector>

#include "sparse_net_global.h"

#include "gen/sparse_net.pb.h"
#include "gen/solution.pb.h"
#include "models/transfer_function.h"
#include "services/synapse_iterator.h"

namespace sparse_net_library {

using std::vector;

class Partial_solution_solver{

public:
  Partial_solution_solver(
    const Partial_solution& partial_solution, 
    uint32 output_layer_first_index_ = UINT32_MAX, 
    Service_context service_context = Service_context()
  ): detail(partial_solution)
  , internal_iterator(detail.inside_indices())
  , input_iterator(detail.input_data())
  , output_iterator(detail.output_data())
  , output_layer_first_index(output_layer_first_index_)
  , neuron_output(detail.internal_neuron_number())
  , collected_input_data(input_iterator.size())
  , transfer_function(service_context)
  { reset(); }

  /**
   * @brief      Gets the size of the elements taken by the configurad Patial solution.
   *
   * @return     The input size in number of elements ( @sdouble32 ).
   */
  uint32 get_input_size(void) const;

  /**
   * @brief      Gives back the size of the array the required Gradient data is stored in.
   *             the gradient data contains intermediate calculations of the output layer neurons, 
   *             which is required to calcualte gradient information.
   *
   * @return     The gradient data array size.
   */
  const uint32 get_gradient_data_size() const{
    if(transfer_function_input.size() == transfer_function_output.size())
      return transfer_function_output.size();
    else throw "Neuron gradient information not consistent!";
  }

  /**
   * @brief      Collects the input stated inside the @Partial_solution into @collected_input_data
   *
   * @param      input_data   The input data
   * @param[in]  neuron_data  The neuron data
   */
  void collect_input_data(vector<sdouble32>& input_data, vector<sdouble32>& neuron_data);

  /**
   * @brief      Provides output data into the given reference
   *
   * @param      neuron_data  The reference to the neuron data
   */
  void provide_output_data(vector<sdouble32>& neuron_data){
    uint32 output_index_start = 0;
    vector<sdouble32> neuron_output_copy(neuron_output);
    output_iterator.skim([&](int synapse_starts, unsigned int synapse_size){
      if(neuron_data.size() < (synapse_starts + synapse_size))
        throw "Neuron data out of Bounds!";
      swap_ranges( /* Save output into the internal neuron memory */
        neuron_output_copy.begin() + output_index_start,
        neuron_output_copy.begin() + output_index_start + synapse_size,
        neuron_data.begin() + synapse_starts
      );
      output_index_start += synapse_size;
    });
  }

  /**
   * @brief      Provides the gradient data to the given references
   *
   * @param      transfer_function_input_   The reference to transfer function input
   * @param      transfer_function_output_  The reference to transfer function output
   */
  void provide_gradient_data(vector<sdouble32>& transfer_function_input_, vector<sdouble32>& transfer_function_output_){
    if(transfer_function_input.size() != transfer_function_output.size()) throw "Neuron gradient data Incompatible!";
    uint32 copy_num = 0, offset = 0;
    output_iterator.skim([&](int synapse_starts, unsigned int synapse_size){
      copy_num = std::min( synapse_size,
        std::max(output_layer_first_index,(synapse_starts + synapse_size)) - output_layer_first_index
      );
      if(0 < copy_num){
        std::copy(
          transfer_function_input.begin() + offset, transfer_function_input.begin() + offset + copy_num,
          transfer_function_input_.begin() + offset
        );
        std::copy(
          transfer_function_output.begin() + offset, transfer_function_output.begin() + offset + copy_num,
          transfer_function_output_.begin() + offset
        );
        offset += copy_num;
      }
    });
  }

  /**
   * @brief      Solves the partial solution in the given argument
   *             to be later supplied through @provide_output_data and @provide_gradient_data
   */
  void solve();

  /**
   * @brief      Resets the data of the included Neurons.
   */
  void reset(void){
    for(sdouble32& neuron_data : neuron_output) neuron_data = 0;
    uint32 index = 1;
    output_iterator.iterate([&](int synapse_index){
      if(static_cast<int>(output_layer_first_index) < synapse_index)
        ++index;
    });
    transfer_function_input = vector<sdouble32>(index,0);
    transfer_function_output = vector<sdouble32>(index,0);
  }

  /**
   * @brief      Determines if given Solution Detail is valid. Due to performance reasons
   *             this function isn't used while solving a SparseNet
   *
   * @return     True if detail is valid, False otherwise.
   */
  bool is_valid(void) const;

private:
  /**
   * The Partial solution to solve
   */
  const Partial_solution& detail;

  /**
   * The iterator to go through the Neurons while solving the detail
   */
  Synapse_iterator internal_iterator;

  /**
   * The iterator to go through the I/O of the detail
   */
  Synapse_iterator input_iterator;
  Synapse_iterator output_iterator;

  /**
   * For Gradient information, intermeidate results are required to be stored.
   * The Partial solution solver shall store the last num_of_transitional_data 
   * of intermediate results in @transfer_function_input and @transfer_function_output
   */
  uint32 output_layer_first_index;
  vector<sdouble32> transfer_function_input;
  vector<sdouble32> transfer_function_output;

  /**
   * The data collected from @Neurons when they are solved
   */
  vector<sdouble32> neuron_output;

  /**
   * The data collected from the @Partial_solution input
   */
  vector<sdouble32> collected_input_data;

  /**
   * The transfer function set configured for the current session
   */
  Transfer_function transfer_function;

};

} /* namespace sparse_net_library */
#endif /* Partial_solution_H */
