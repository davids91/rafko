#include "services/neuron_router.h"

#include <algorithm>
#include <thread>

#include "models/neuron_info.h"
#include "services/synapse_iterator.h"

namespace sparse_net_library{

using std::move;
using std::swap;
using std::thread;

Neuron_router::Neuron_router(SparseNet& sparse_net) : net(sparse_net){
  output_layer_iterator = (net.neuron_array_size() - net.output_neuron_number()); /* Start to process Ouptut Layer Neurons */
  neuron_states = vector<unique_ptr<atomic<uint32>>>(); /* Every Neuron has 0 child processed at first */
  neuron_number_of_inputs = vector<uint32>(net.neuron_array_size());

  for(int neuron_iterator; neuron_iterator < net.neuron_array_size(); ++neuron_iterator){
      for(int synapse_iterator = 0;
        synapse_iterator < net.neuron_array(neuron_iterator).input_index_sizes_size();
        ++synapse_iterator
      ) neuron_number_of_inputs[neuron_iterator] += net.neuron_array(neuron_iterator).input_index_sizes(synapse_iterator);

      neuron_states.push_back(std::make_unique<atomic<uint32>>());
  } /* Calculating how many children one Neuron has */
  net_subset_size = 0.0;
  net_subset = std::deque<uint32>();
  net_subset_index = std::deque<uint32>();
}

void swap_neuron_routers(Neuron_router& first, Neuron_router& second) noexcept{
  first.net = second.net;
  first.output_layer_iterator = second.output_layer_iterator.load();
  swap(first.neuron_number_of_inputs,second.neuron_number_of_inputs);
  swap(first.neuron_states,second.neuron_states);
  first.net_subset_size = second.net_subset_size.load();
  swap(first.net_subset,second.net_subset);
  swap(first.net_subset_index,second.net_subset_index);
}

Neuron_router::Neuron_router(const Neuron_router& other): net(other.net){
  output_layer_iterator = other.output_layer_iterator.load();
  neuron_number_of_inputs = other.neuron_number_of_inputs;
  neuron_states = vector<unique_ptr<atomic<uint32>>>();
  for(uint32 i = 0; i < other.neuron_states.size(); ++i){
    neuron_states.push_back(std::make_unique<atomic<uint32>>());
    *neuron_states[i] = other.neuron_states[i]->load();
  }
  net_subset = other.net_subset;
  net_subset_index = other.net_subset_index;
}

Neuron_router::Neuron_router(Neuron_router&& other): net(other.net){
  swap_neuron_routers(*this, other);
}

Neuron_router& Neuron_router::operator=(Neuron_router other){
  swap_neuron_routers(*this, other);
  return *this;
}

Neuron_router& Neuron_router::operator=(Neuron_router&& other){
  swap_neuron_routers(*this, other);
  return *this;
}

Neuron_router::~Neuron_router(){
  neuron_number_of_inputs.clear();
  neuron_states.clear();
  net_subset.clear();
  net_subset_index.clear();
}

void Neuron_router::collect_subset(uint16 iteration, uint8 arg_max_solve_threads, sdouble32 arg_device_max_megabytes){
    vector<thread> processing_threads;
    for(uint8 thread_iterator = 0; thread_iterator < arg_max_solve_threads; thread_iterator++){
      processing_threads.push_back(
        thread(
          &Neuron_router::collect_subset_thread,
          this, iteration, arg_max_solve_threads, arg_device_max_megabytes, thread_iterator
        )
      ); /* Add threads for processing */
    }

    std::for_each(processing_threads.begin(),processing_threads.end(),[](thread& processing_thread){
      if(true == processing_thread.joinable())processing_thread.join();
    });
}

void Neuron_router::collect_subset_thread(uint16 iteration, uint8 arg_max_solve_threads, sdouble32 arg_device_max_megabytes, uint8 thread_index){
  /**
   * In order of the iteration, the visited neuron indexes. The First Index is always one of the Output Layer Neurons
   */
  vector<uint32> visiting(1,
    (output_layer_iterator + ((net.neuron_array_size()-1-output_layer_iterator)/arg_max_solve_threads)*thread_index)
  ); /* The first Neuron to be visited is decided based on the number of threads, to make sure the threads are as independent as possible */
  uint32 visiting_next = visiting.back();

  while( /* Iterate the Net until every possible Neuron is collected into an independent subset of it */
    (net.neuron_array_size() > static_cast<int>(visiting.back())) /* The currently visiting Neuron is inside bounds of the net */
    &&(static_cast<int>(output_layer_iterator) < net.neuron_array_size()) /* Until the whole output layer is processed */
    &&(net_subset_size < arg_device_max_megabytes) /* Or there is enough collected Neurons for a Partial solution */
  ){
    visiting_next = get_next_neuron(visiting, iteration);
    add_neuron_into_subset(visiting, visiting_next, iteration);
    step(visiting, visiting_next, iteration);
  }
}

uint32 Neuron_router::get_next_neuron(vector<uint32>& visiting, uint16& iteration){
  uint32 number_of_processed_inputs = 0;
  uint32 expected_number_of_processed_inputs = 0;
  uint32 visiting_next = 0;

  visiting_next = visiting.back();
  while(/* Checking current Neuron and its inputs */
    (is_neuron_subset_candidate(visiting.back(),iteration))
    &&(number_of_processed_inputs < neuron_number_of_inputs[visiting.back()]) /* Neuron has some unprocessed and not reserved inputs */
    &&(visiting.back() == visiting_next)  /* no children are found to move on to */
  ){
    number_of_processed_inputs = 0;
    expected_number_of_processed_inputs = *neuron_states[visiting.back()];
    /*!#4 Number of processed inputs could be used to not start the iteration from the beginning */
    Synapse_iterator iter(net.neuron_array(visiting.back()).input_index_starts(),net.neuron_array(visiting.back()).input_index_sizes());
    iter.iterate([&](int synapse_input_index){
      if(
        (Synapse_iterator::is_index_input(synapse_input_index))
        ||(is_neuron_processed(synapse_input_index))
      ){
        ++number_of_processed_inputs;
      }else if(
        (!Synapse_iterator::is_index_input(synapse_input_index))
        &&(is_neuron_subset_candidate(synapse_input_index, iteration))
      ){
        visiting_next = synapse_input_index;
        return  false;
      }
      return true;
    });
    if( /* Some inputs are still unprocessed */
      (number_of_processed_inputs < neuron_number_of_inputs[visiting.back()])
      &&(visiting_next == visiting.back()) /* There are no next input to iterate to */
    ){
      (void)neuron_states[visiting.back()]->compare_exchange_strong(
        expected_number_of_processed_inputs,
        neuron_state_next_iteration_value(visiting.back(),iteration)
      ); /* If another thread updated the Neuron status before this one, that's fine too */
    }else{ /* Neuron has unprocessed inputs still, iteration shall continue with one of them */
      (void)neuron_states[visiting.back()]->compare_exchange_strong(
        expected_number_of_processed_inputs,
        number_of_processed_inputs
      ); /* If another thread updated the Neuron status before this one, that's fine too */
    }
  } /* Checking current Neuron and its inputs */
  return visiting_next;
}

void Neuron_router::add_neuron_into_subset(vector<uint32>& visiting, uint32& visiting_next, uint16& iteration){
  uint32 tmp_index = 0;
  sdouble32 tmp_size = 0;

  if(
    (is_neuron_solvable(visiting.back()))
    &&(is_neuron_subset_candidate(visiting.back(),iteration))
    &&(visiting_next == visiting.back()) /* There are no free input neurons to iterate to */
    &&((neuron_states[visiting.back()])->compare_exchange_strong(neuron_number_of_inputs[visiting.back()],neuron_state_reserved_value(visiting.back())))
  ){ /* Able to lock a relevant Neuron, Include it into the Solution! */
    std::lock_guard<std::mutex> lock(net_subset_mutex);
    net_subset.push_back(visiting.back());
    net_subset_index.push_back(std::numeric_limits<uint32>::max());
    net_subset_mutex.unlock();

    /* Collect estimated size of Neuron in the @Partial_solution */
    tmp_index = Neuron_info::get_neuron_estimated_size_bytes(net.neuron_array(visiting.back()));
    tmp_size = net_subset_size; /* Add estimated Neuron Size */
    while(!net_subset_size.compare_exchange_weak(
      tmp_size,tmp_size + static_cast<sdouble32>(tmp_index)/(1024.0 * 1024.0)
    ))tmp_size = net_subset_size;
  } /* No matter if the lock fails or the Neuron has reserved inputs */
}

void Neuron_router::step(vector<uint32>& visiting, uint32& visiting_next, uint16& iteration){
    uint32 tmp_index = 0;

    if(visiting_next != visiting.back()){ /* found another Neuron to iterate to because the index values differ (because visiting_next is updated!) */
      visiting.push_back(visiting_next);
    }else if(1 < visiting.size()){ /* haven't found another Neuron to iterate to, try with parent Neuron, if there is any */
      visiting.pop_back(); /* remove latest Neuron from the queue, go to its parent in the next iteration */
    }
    if(1 == visiting.size()){ /* The Visiting vector is down to it's last element, which is the visit-starting output layer neuron */
      tmp_index = visiting.back();
      if((!is_neuron_in_progress(tmp_index))&&(!is_neuron_subset_candidate(tmp_index, iteration)))
        visiting.back()++; /* If Neuron is processed, reserved or not relevant to the current iteration go to the next one */
      /*!Note: It is possible to get out of bounds here, it will mean that this thread is finished, and collection ( if needed ) will restart in the next iteration */
      if(
        (is_neuron_processed(tmp_index))
        &&(tmp_index == output_layer_iterator) /* If the Neuron at @output_layer_iterator is processed */
        &&(static_cast<int>(output_layer_iterator) < (net.neuron_array_size()-1)) /* And it shall remain in bounds of the array */
      ){ /*  step the output_layer_iterator forward! */
        (void)output_layer_iterator.compare_exchange_strong(tmp_index, tmp_index+1 );
        /*!Note: @output_layer_iterator may have been updated within another thread, but that's okay */
      }
    } /* (1 == visiting.size()) */
}

void Neuron_router::set_neuron_to_processed(uint32 neuron_index){
  uint32 tmp = neuron_state_reserved_value(neuron_index);
  (neuron_states[neuron_index])->compare_exchange_strong(tmp, neuron_state_processed_value(neuron_index));
}

bool Neuron_router::get_first_neuron_index_from_subset(uint32& put_it_here){
  bool ret = false;
  if(0 < net_subset.size()){
    put_it_here = net_subset.front();
    ret = true;
  }
  return ret;
}

bool Neuron_router::confirm_first_subset_element_processed(uint32 neuron_index){
  bool ret = false;
  if((0 < net_subset.size())&&(neuron_index == net_subset.front())){
    set_neuron_to_processed(neuron_index);
    net_subset.pop_front();
    ret = true;
  }
  return ret;
}

void Neuron_router::correct_subset_memory_usage(sdouble32 memory_delta){
    sdouble32 tmp_size = net_subset_size;
    
    if(tmp_size > memory_delta)
      throw "Can't correct memory usage below 0!";

    while(!net_subset_size.compare_exchange_weak(
      tmp_size,tmp_size - memory_delta
    ))tmp_size = net_subset_size;
}

} /* namespace sparse_net_library */
