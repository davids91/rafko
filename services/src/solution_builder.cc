#include "services/solution_builder.h"


#include "models/neuron_info.h"

namespace sparse_net_library{

Solution_builder& Solution_builder::max_solve_threads(uint8 number){
  arg_max_solve_threads = number;
  is_max_solve_threads_set = true;
  return *this;
}

Solution_builder& Solution_builder::device_max_megabytes(sdouble32 megabytes){
  arg_device_max_megabytes = megabytes;
  return *this;
}

Solution_builder& Solution_builder::arena_ptr(google::protobuf::Arena* arena){
  arg_arena_ptr = arena;
  return *this;
}

Solution* Solution_builder::build( SparseNet& net ){
  Partial_solution* current_partial = google::protobuf::Arena::CreateMessage<Partial_solution>(arg_arena_ptr);
  vector<vector<Partial_solution*>> partial_matrix;
  uint16 iteration = 1; /* Has to start with 1, otherwise values mix with neuron processed value */
  Neuron_router net_iterator = Neuron_router(net);

  while(!net_iterator.finished()){ /* Until the whole output layer is processed */
    net_iterator.collect_subset(iteration,arg_max_solve_threads,arg_device_max_megabytes);
    generate_partial_solution_from_subset(net,net_iterator,*current_partial);
    ++iteration;
  }

  /* #######################################################################################################
   * Build a @Solution from the @Partial_Solution Vector
   * ####################################################################################################### */
  //Solution* full_solution = google::protobuf::Arena::CreateMessage<Solution>(arg_arena_ptr);
  /*!#4 Use the Partial solutions matrix here */

  /* Build Solution from processed net */
  throw "Solution builder not fully implemented yet..";
}

void Solution_builder::generate_partial_solution_from_subset(const SparseNet& net, Neuron_router& net_iterator, Partial_solution& current_partial){
  /* #######################################################################################################
   * Update Neuron state to processed ( temporary )
   * ####################################################################################################### */
  uint32 neuron_index;
  sdouble32 memory_usage = 0;
  while(
    (memory_usage < arg_device_max_megabytes)
    &&(net_iterator.get_first_neuron_index_from_subset(neuron_index))
  ){
    add_neuron_to_partial_solution(net, neuron_index, current_partial);
    net_iterator.confirm_first_subset_element_processed(neuron_index);
  }

  /*!#4 correct memory usage */

}

bool Solution_builder::add_neuron_to_partial_solution(const SparseNet& net, uint32 neuron_index, Partial_solution& partial){
  if(net.neuron_array_size() > static_cast<int>(neuron_index)){
#if 0 /*!#4 Needs to sort out Partial Solution inputs first */
    const Neuron& neuron = net.neuron_array(neuron_index);
    uint32 input_size = 0;

    /* Add a new Neuron into the partial solution */
    partial_set_internal_neuron_number(partial.internal_neuron_number() + 1);
    partial.add_actual_index(neuron_index);

    /* Copy in Neuron parameters */
    partial.add_neuron_transfer_function(neuron.transfer_function_idx());
    partial.add_memory_ratio_index(partial.weight_table_size());
    partial.add_weight_table(net.weight_table(neuron.memory_ratio_idx()));
    partial.add_bias_index(partial.weight_table_size());
    partial.add_weight_table(net.weight_table(neuron.bias_idx()));

    /* Copy in weights from the net*/
    partial.add_weight_synapse_number(neuron.weight_index_starts_size());
    for(uint32 weight_synapse_iterator = 0; weight_synapse_iterator < neuron.weight_index_starts_size(); ++weight_synapse_iterator){
      partial.add_weight_index_starts(partial.weight_table_size());
      partial.add_weight_index_sizes(neuron.weight_index_sizes(weight_synapse));
      for(uint32 weight_iterator = 0; weight_iterator < neuron.weight_index_size(weight_synapse_iterator); ++weight_iterator){
        partial.add_weight_table(net.weight_table(neuron.weight_index_starts(weight_synapse_iterator) + weight_iterator));
      } /* For Every weight inside a synapse */
    } /* For every weight synapse inside a Neuron */

    /* Copy in input data references */
    /*!#4 Input data or input Neuron? how to differentiate? */
    //partial.input_data_size();

    /* Copy in input indexes */
    //partial.add_index_synapse_number(neuron.input_index_starts_size());
#endif
    return true;
  }else throw "Neuron index is out of bounds from net neuron array!";
}

} /* namespace sparse_net_library */
