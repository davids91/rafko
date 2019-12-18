#include <thread>

#include "services/solution_builder.h"

namespace sparse_net_library{

  using std::thread;

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
  /* Define resources to be built */
  Solution* full_solution = google::protobuf::Arena::CreateMessage<Solution>(arg_arena_ptr);
  //!#4 content = google::protobuf::Arena::CreateMessage<Decoupled_solutions>(arg_arena_ptr);
  vector<thread> processing_threads;

  /* #######################################################################################################
   * Set Helper variables
   * ####################################################################################################### */
  output_layer_iterator = (net.neuron_array_size() - net.output_neuron_number()); /* Start to process Ouptut Layer Neurons */
  neuron_states = vector<unique_ptr<atomic<uint32>>>(net.neuron_array_size()); /* Every Neuron has 0 child processed at first */
  neuron_number_of_inputs = vector<uint32>(net.neuron_array_size());
  net_subset_size = 0.0;
  net_subset = std::deque<sdouble32>();
  net_subset_index = std::deque<uint32>();
  for(int neuron_iterator; neuron_iterator < net.neuron_array_size(); ++neuron_iterator){
      for(int partition_iterator = 0;
        partition_iterator < net.neuron_array(neuron_iterator).input_index_sizes_size();
        ++partition_iterator
      ){
        neuron_number_of_inputs[neuron_iterator] += net.neuron_array(neuron_iterator).input_index_sizes(partition_iterator);
      }

      if(neuron_iterator < static_cast<int>(net.input_neuron_number())){ /* Neuron connected to the input directly */
        *neuron_states[neuron_iterator] = neuron_number_of_inputs[neuron_iterator]; /* The inputs already count as "solvable" */
      }
  } /* Calculating how many children one Neuron has */

  while(static_cast<int>(output_layer_iterator) < net.neuron_array_size()){ /* Until the whole output layer is processed */
    /* #######################################################################################################
     * Extend the net_subset until memory limit reached or the net is complete
     * ####################################################################################################### */
    for(uint8 thread_iterator = 0; thread_iterator < arg_max_solve_threads; thread_iterator++){
      processing_threads.push_back(thread(&Solution_builder::collect_subset_thread, this, std::ref(net), std::ref(*full_solution), thread_iterator)); /* Add threads for processing */
    }

    /* Wait for every Thread to finish */
    std::for_each(processing_threads.begin(),processing_threads.end(),[](thread& processing_thread){
      if(true == processing_thread.joinable())
        processing_thread.join();
    });
    /* #######################################################################################################
     * Start to build a @Partial_solution from the subset; correct memory usage
     * ####################################################################################################### */
    /*!#4 Create the Partial solution from the subset */
    /*!#4 Construct the solution from the collected Vector */
    get_partial_solution_from_subset(net,net_subset);
    /* #######################################################################################################
     * Construct somepartial solutions from the collected Subsets
     * ####################################################################################################### */
    /* #######################################################################################################
     * If Partial_solution is complete update Neuron states
     * - size of it is at the memory limit, or the net is complete
     * ####################################################################################################### */
  }
  neuron_states.clear(); /* Clean up atomic numbers */


  /* Build Solution from processed net */
  throw NOT_IMPLEMENTED_EXCEPTION;
}

void Solution_builder::collect_subset_thread( SparseNet& net, Solution& result, uint8 thread_index){
  /* #######################################################################################################
   * Local Iterators
   * ####################################################################################################### */
  /**
   * In order of the iteration, the visited neuron indexes. The First Index is always one of the Output Layer Neurons
   */
  vector<uint32> visiting(1,
    (output_layer_iterator + ((net.neuron_array_size()-output_layer_iterator)/arg_max_solve_threads)*thread_index)
  ); /* The first Neuron to be visited is decided based on the number of threads, to make sure the threads are as independent as possible */
  uint32 visiting_next = visiting.back();
  uint32 number_of_processed_inputs;
  uint32 expected_number_of_processed_inputs;
  uint32 tmp_index;

  /* #######################################################################################################
   * Iterating the net
   * ####################################################################################################### */
  if(net.neuron_array_size() > static_cast<int>(visiting.back())){ /* The currently visiting Neuron is inside bounds of the net */
    while(
      (static_cast<int>(output_layer_iterator) < net.neuron_array_size()){ /* Until the whole output layer is processed */
      &&(net_subset_size < arg_device_max_megabytes) /* Or there is enough collected Neurons for a Partial solution */
    )
      visiting_next = visiting.back(); /* If visiting.back() == visiting_next it means no children are found to move on to */

      if(is_neuron_in_progress(visiting.back())){
        /* #######################################################################################################
         * Checking current Neuron and its inputs
         * ####################################################################################################### */
        while(
          is_neuron_in_progress(visiting.back()) /* every input is processed */
          &&((number_of_processed_inputs + number_of_reserved_inputs) < neuron_number_of_inputs[visiting.back()]) /* Neuron has some unprocessed and not reserved inputs */
          &&(visiting.back() == visiting_next) /* no neuron found the iteration can continue with */
        ){
          if(visiting.back() >= net.input_neuron_number()){ /* Neuron not connected to the input directly */
            number_of_processed_inputs = 0;
            expected_number_of_processed_inputs = *neuron_states[visiting.back()];
            /*! Number of processed inputs could be used to not start the iteration from the beginning */
            for(
              int partition_iterator=0; /* iterate through the Neuron partitions */
              partition_iterator < net.neuron_array(visiting.back()).input_index_sizes_size();
              ++partition_iterator
            ){
              for(
                uint32 input_interator=0; /* iterate through the partition inputs */
                input_interator < net.neuron_array(visiting.back()).input_index_sizes(partition_iterator);
                ++input_interator
              ){
                tmp_index = net.neuron_array(visiting.back()).input_index_starts(partition_iterator) + input_interator;
                if(is_neuron_processed(tmp_index)){
                  number_of_processed_inputs ++; /* ??? */
                }else if(!is_neuron_reserved(tmp_index)){
                  visiting_next = tmp_index;
                  goto end_of_neuron_input_loop; /* don't judge */
                }else number_of_reserved_inputs++;
              }/* iterate through the partition inputs */
            } /* iterate through the Neuron partitions */
            end_of_neuron_input_loop:
            (void)neuron_states[visiting.back()]->compare_exchange_strong(
              expected_number_of_processed_inputs,
              number_of_processed_inputs
            ); /* If another thread updated the Neuron status before this one, that's fine too */
          } /*! Neuron not connected to the input directly: else block shouldn't be active because
             * at pre-processing the neuron_states are set to all children processed in the input layer */
        } /* Until every input is processed */

        /* #######################################################################################################
         * Add the Neuron into the current subset of the net
         * ####################################################################################################### */
        if(
          (visiting_next == visiting.back()) /* There are no free input neurons to iterate to */
          &&(is_neuron_solvable(visiting.back()))
          &&((neuron_states[visiting.back()])->compare_exchange_strong(
                number_of_processed_inputs,
                number_of_processed_inputs + 1
          )) /* was able to lock the Neuron successfully */
        ){ /* Include the Neuron into the Solution! */
            std::lock_guard<std::mutex> lock(net_subset_mutex);
            net_subset.push_back(visiting.back());
            net_subset_index.push_back(std::numeric_limits<uint32>::max());
            net_subset_mutex.unlock();

            /* Collect estimated size of Neuron in the @Partial_solution */
            tmp_index = neuron_number_of_inputs[visiting.back()] * 4/* Bytes */ * 2/* field (weights and inputs) */
            tmp_index += net->neuron_array(visiting.back()).input_index_starts_size() * 2/* Byte */ * 2/* field ( size and starts) */;
            net_subset_size.fetch_adds(tatic_cast<sdouble32>(tmp_index)/(1024.0 * 1024.0);/* Add partition starts sum size */

        } /* No matter if the lock fails or the Neuron has reserved inputs */
      }/*(is_neuron_in_progress(visiting.back()))*/

      /* #######################################################################################################
       * Decide the next Neuron to iterate to and increase the output layer iterator
       * ####################################################################################################### */
      if(visiting_next != visiting.back()){ /* found another Neuron to iterate to because the index values differ (because visiting_next is updated!) */
          visiting.back() = visiting_next;
      }else if(1 < visiting.size()){ /* haven't found another Neuron to iterate to, try with parent Neuron */
          visiting.pop_back(); /* remove latest Neuron from the queue, go to its parent in the next iteration */
      }

      if(1 == visiting.size()){ /* The Visiting vector is down to it's last element, which is the visit-starting output layer neuron */
        tmp_index = visiting.back();
        if((net.neuron_array_size()-1u) > tmp_index){ /* current is not the last Neuron in the net */
          if(!is_neuron_in_progress(tmp_index))visiting.back()++; /* If Neuron is processed, or reserved, go to the next one */
        }else{ /* Iteration is at the last neuron in the net */
          if(
            !is_neuron_in_progress(tmp_index) /* Neuron is finished */
            &&(output_layer_iterator != tmp_index) /* And it's not the output layer iterator */
          ) visiting.back() = output_layer_iterator;
        }

        if(is_neuron_processed(tmp_index)&&(tmp_index == output_layer_iterator)){ /* If the Neuron at @output_layer_iterator is processed in this iteration */
          if(!output_layer_iterator.compare_exchange_strong(tmp_index, tmp_index+1 ) ){ /* @output_layer_iterator have been updated within another thread */
            visiting.back() = std::min( /* Overwrite previously given index value to minimize collisions inbetween threads */
              static_cast<uint32>(net.neuron_array_size() - 1), /* At maximum is the last Neuron */
              (output_layer_iterator + (((net.neuron_array_size()-output_layer_iterator)/arg_max_solve_threads)*thread_index))
            ); /* Otherwise go to an Output Layer Neuron based on the thread index */
          }
        }
      } /* (1 == visiting.size()) */
    }/* while(output_layer_iterator < net.neuron_array_size()) */
  }else throw INVALID_NET_EXCEPTION;
}

unique_ptr<Decoupled_solutions> Solution_builder::get_partial_solution_from_subset( SparseNet& net, deque<uint32>& net_subset){
  uint32 tmp_index;
  for(uint32 neuron_index : net_subset){
    tmp_index = neuron_number_of_inputs[neuron_index] + 2;
    (neuron_states[neuron_index])->compare_exchange_strong(
       tmp_index, tmp_index - 2
    );
  }
  return nullptr;
}

} /* namespace sparse_net_library */
