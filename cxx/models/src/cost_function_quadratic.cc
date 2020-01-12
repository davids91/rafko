#include "models/cost_function_quadratic.h"

#include <atomic>
#include <algorithm>

namespace sparse_net_library{

sdouble32 Cost_function_quadratic::get_error() const{
  if(features.size() == labels.size()){
    sdouble32 score = 0;
    uint32 feature_iterator = 0;
    uint32 feature_size = features[0].size();
    
    while(feature_iterator < feature_size){ /* evaluate a feature(Neuron output) on all samples */
      score += double_error_for_feature(feature_iterator);
      ++feature_iterator;
    }

    return (0.5 * score);
  }else throw "Incompatible Feature and Label sizes!";
}

sdouble32 Cost_function_quadratic::get_error(uint32 feature_index) const{
  if(/* Check if input index is valid */
    (features.size() > feature_index)
    &&(labels.size() > feature_index)
    &&(features[feature_index].size() == labels[feature_index].size())
  ){
    return (0.5 * double_error_for_feature(feature_index));
  }else throw "Incompatible Feature and Label sizes!";
}

sdouble32 Cost_function_quadratic::double_error_for_feature(uint32 feature_index) const{

  using std::thread;
  using std::future;
  using std::ref;

  uint32 minibatch_size = features.size();
  uint32 sample_iterator = 0;
  uint32 thread_iterator;
  atomic<sdouble32> score;
  vector<future<sdouble32>> promises = vector<future<sdouble32>>();
  mutex promises_mutex;
  thread collector_thread( &Cost_function_quadratic::collect_promises, 
    this, ref(promises), ref(score), ref(promises_mutex)
  );

  score.store(0);
  while(sample_iterator < minibatch_size){
    thread_iterator = 0;
    while(promises.size() < max_threads);
    lock_guard<mutex> my_lock(promises_mutex);
    promises.push_back(std::async(
      &Cost_function_quadratic::sample_distance_squared, 
      this, feature_index, (sample_iterator + thread_iterator)
    ));
    ++sample_iterator;
  }
  collector_thread.~thread();
  return (score);
}

} /* namespace sparse_net_library */
