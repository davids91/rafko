#ifndef WEIGHT_UPDATER_H
#define WEIGHT_UPDATER_H

#include "sparse_net_global.h"
#include "gen/common.pb.h"

namespace sparse_net_library{

using std::atomic;
using std::thread;
using std::ref;

/**
 * @brief      Base implementation for updating weights for netowrks based on weight gradients
 */
class Weight_updater{
public:
  Weight_updater(
    SparseNet& sparse_net, vector<unique_ptr<atomic<sdouble32>>>& weight_gradients_, 
    Service_context& context
  ): net(sparse_net)
  , service_context(context)
  , weight_gradients(weight_gradients_)
  , calculate_threads(0)
  { 
    calculate_threads.reserve(context.get_max_processing_threads());
  };

  void update_weights_with_gradients(void){
    uint32 process_thread_iterator = 0;
    while(static_cast<int>(process_thread_iterator) < net.weight_table_size()){
      while(
        (service_context.get_max_processing_threads() > calculate_threads.size())
        &&(net.weight_table_size() > static_cast<int>(process_thread_iterator))
      ){
        calculate_threads.push_back(
          thread(&Weight_updater::update_weight_with_gradient, this, process_thread_iterator)
        );
        ++process_thread_iterator;
      }
      while(0 < calculate_threads.size())
        if(calculate_threads.back().joinable()){
          calculate_threads.back().join();
          calculate_threads.pop_back();
        }
    }
  }

private:
  SparseNet& net;
  Service_context& service_context;
  vector<unique_ptr<atomic<sdouble32>>>& weight_gradients;
  vector<thread> calculate_threads;

  void update_weight_with_gradient(uint32 weight_index){
    net.set_weight_table( weight_index,
      net.weight_table(weight_index) + *weight_gradients[weight_index] * service_context.get_step_size()
    );

  }
};

} /* namespace sparse_net_library */

#endif /* WEIGHT_UPDATER_H */