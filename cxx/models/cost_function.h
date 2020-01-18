#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include "sparse_net_global.h"

#include <vector>
#include <thread>
#include <mutex>
#include <future>
#include <functional>

namespace sparse_net_library{

using std::vector;
using std::atomic;
using std::mutex;
using std::future;
using std::lock_guard;
using std::function;

/**
 * @brief      Error function handling and utilities, provides a hook for a computation 
 *             function to be run on every sample by feature.
 */
class Cost_function{
public:
  Cost_function(uint8 maximum_threads, vector<vector<sdouble32>>& feature_samples, vector<vector<sdouble32>>& label_samples)
  : max_threads(maximum_threads), features(feature_samples), labels(label_samples){};

  /**
   * @brief      Gets the overall error for a given feature and labelset
   *
   * @return     The error.
   */
  virtual sdouble32 get_error() const = 0;

  /**
   * @brief      Gets the error for a feature for every sample
   *
   * @param[in]  feature_index  The feature index
   *
   * @return     The error.
   */
  virtual sdouble32 get_error(uint32 feature_index) const = 0;

  /**
   * @brief      Gets the the Cost function derivative to a feature
   *
   * @param[in]  feature_index  The index of the feature
   *
   * @return     The d cost over d feature.
   */
  virtual sdouble32 get_d_cost_over_d_feature(uint32 feature_index) const = 0;

protected:
  uint8 max_threads;
  vector<vector<sdouble32>>& features;
  vector<vector<sdouble32>>& labels;

  /**
   * @brief      Throws an exception if the references are incorrectly set up
   */
  void verify_sizes() const{
    if(features.size() != labels.size())
    throw "Incompatible Feature and Label sizes!";
  }

  /**
   * @brief      Throws an exception if the references for a given features are incorrectly set up
   *
   * @param[in]  feature_index  The feature index
   */
  void verify_sizes(uint32 feature_index) const{
    if(/* Check if input index is valid */
      (features.size() <= feature_index)
      ||(labels.size() <= feature_index)
      ||(features[feature_index].size() != labels[feature_index].size())
    )throw "Incompatible Feature and Label sizes!";
  }

  /**
   * @brief      Utility function to run a calculation on a feature for every sample
   *
   * @param[in]  feature_index  The index of the feature to calculate on
   * @param[in]  calculation    The calculation to get the error uder the given feature index
   *
   * @return     the result of the given calculation
   */
  sdouble32 calculate_for_feature(uint32 feature_index, function<sdouble32(sdouble32,sdouble32)> calculation) const{
      using std::thread;
      using std::ref;
      vector<future<sdouble32>> promises = vector<future<sdouble32>>(); /* The results of the active threads */
      mutex promises_mutex;
      bool in_progress = true;
      thread initiator_thread( &Cost_function::initiate_promises,
        this, ref(promises), ref(promises_mutex), feature_index, calculation );
      future<sdouble32> score = std::async( &Cost_function::collect_promises,
        this, ref(promises), ref(promises_mutex), ref(in_progress) );
      initiator_thread.join();
      while(in_progress){
        lock_guard<mutex> my_lock(promises_mutex);
        in_progress = (0 < promises.size());
      } /* Wait until all of the calculations finish */
      score.wait();
      return (score.get());
  }

  /**
   * @brief      The thread to initiate calculations for every feature-label pair on every sample,
   *             and push them into the promises array
   *
   * @param      promises        The promises
   * @param      promises_mutex  The promises mutex
   * @param[in]  feature_index   The feature index
   * @param[in]  used_function   The used function
   */
  void initiate_promises(vector<future<sdouble32>>& promises,
      mutex& promises_mutex, uint32 feature_index, function<sdouble32(sdouble32,sdouble32)> used_function) const{
    uint32 minibatch_size = features.size();
    uint32 sample_iterator = 0;
    while(sample_iterator < minibatch_size){
      while(promises.size() > max_threads);
      lock_guard<mutex> my_lock(promises_mutex);
      while((promises.size() < max_threads)&&(sample_iterator < minibatch_size)){
        promises.push_back(std::async(
          used_function, features[sample_iterator][feature_index], labels[sample_iterator][feature_index]
        ));
        ++sample_iterator;
      } /* while(promises.size() < max_threads)&&(sample_iterator < minibatch_size) */
    } /* while(sample_iterator < minibatch_size) */
  }

  /**
   * @brief      The thread to collect pending caluclations and update the promises array accordingly
   *
   * @param      promises        The promises
   * @param      score           The score
   * @param      promises_mutex  The promises mutex
   * @param      in_progress     In progress
   */
  sdouble32 collect_promises(vector<future<sdouble32>>& promises, mutex& promises_mutex, bool& in_progress) const{
    sdouble32 score = 0;
    while(in_progress){
      lock_guard<mutex> my_lock(promises_mutex);
      if(0 < promises.size()){
        score += promises.front().get();
        promises.erase(promises.begin());
      }
    } /* while(in_progress) */
    return score;
  }
};

} /* namespace sparse_net_library */
#endif /* COST_FUNCTION_H */
