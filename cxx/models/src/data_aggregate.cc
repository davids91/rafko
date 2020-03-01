#include "models/data_aggregate.h"

namespace sparse_net_library{

void Data_aggregate::fill(Data_set& samples){
  uint32 feature_start_index = 0;
  for(sint32 sample_iterator = 0; sample_iterator < samples.features_size(); ++ sample_iterator){
    input_samples[sample_iterator] = vector<sdouble32>(samples.feature_size());
    label_samples[sample_iterator] = vector<sdouble32>(samples.feature_size());
    for(uint32 feature_iterator = 0; feature_iterator < samples.feature_size() ; ++feature_iterator){
      input_samples[sample_iterator][feature_iterator] = samples.features(feature_start_index + feature_iterator);
      label_samples[sample_iterator][feature_iterator] = samples.labels(feature_start_index + feature_iterator);
    }
  }
}

void Data_aggregate::set_feature_for_label(uint32 sample_index, const vector<sdouble32>& neuron_data){
  sdouble32 buffer;
  sdouble32 local_error = sample_errors[sample_index] / static_cast<sdouble32>(get_number_of_samples());
  if(label_samples.size() > sample_index){
    while(average_error.compare_exchange_weak(buffer,(buffer - local_error)))buffer = average_error;
    sample_errors[sample_index] = cost_function.get_error(label_samples[sample_index], neuron_data);
    local_error = sample_errors[sample_index] / static_cast<sdouble32>(get_number_of_samples());
    while(average_error.compare_exchange_weak(buffer,(buffer + local_error)))buffer = average_error;
  }else throw "Sample index out of bounds!";
}

} /* namespace sparse_net_library*/
