#ifndef COST_FUNCTION_QUADRATIC_H
#define COST_FUNCTION_QUADRATIC_H

#include "models/cost_function.h"

#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <future>

namespace sparse_net_library{

using std::vector;
using std::atomic;
using std::mutex;
using std::future;
using std::lock_guard;

/**
 * @brief      Error function handling and utilities
 */
class Cost_function_quadratic : public Cost_function{
public:
 Cost_function_quadratic(uint8 maximum_threads, vector<vector<sdouble32>>& feature_samples, vector<vector<sdouble32>>& label_samples)
 : Cost_function(maximum_threads, feature_samples, label_samples){}
  sdouble32 get_error() const;
  sdouble32 get_error(uint32 feature_index) const;

private:
  void collect_promises(vector<future<sdouble32>>& promises,
      atomic<sdouble32>& score, mutex& promises_mutex, bool& in_progress) const{
    sdouble32 temp;
    sdouble32 buffer;
    while(in_progress){
      lock_guard<mutex> my_lock(promises_mutex);
      if(0 < promises.size()){
        temp = score + promises.front().get();
        buffer = score;
        while(!score.compare_exchange_weak(buffer, temp))buffer = score;
        promises.erase(promises.begin());
      }
    }
  }
  sdouble32 feature_distance_squared_sum(uint32 feature_index) const;
  sdouble32 sample_distance_squared(uint32 feature_index, uint32 sample_index) const{
    return pow((features[sample_index][feature_index] - labels[sample_index][feature_index]),2.0);
  }
};

} /* namespace sparse_net_library */
#endif /* COST_FUNCTION_QUADRATIC_H */
