#include "models/data_aggregate.h"

namespace sparse_net_library{

void Data_aggregate::fill(Data_set& samples){
  uint32 feature_start_index = 0;
  uint32 input_start_index = 0;
  for(uint32 sample_iterator = 0; sample_iterator < get_number_of_samples(); ++ sample_iterator){
    input_samples[sample_iterator] = vector<sdouble32>(samples.input_size());
    label_samples[sample_iterator] = vector<sdouble32>(samples.feature_size());
    for(uint32 input_iterator = 0; input_iterator < samples.input_size(); ++input_iterator)
      input_samples[sample_iterator][input_iterator] = samples.inputs(input_start_index + input_iterator);
    for(uint32 feature_iterator = 0; feature_iterator < samples.feature_size(); ++feature_iterator)
      label_samples[sample_iterator][feature_iterator] = samples.labels(feature_start_index + feature_iterator);
    input_start_index += samples.input_size();
    feature_start_index += samples.feature_size();
  }
}

void Data_aggregate::set_feature_for_label(uint32 sample_index, const vector<sdouble32>& neuron_data){
  sdouble32 buffer = error_sum;
  if(label_samples.size() > sample_index){
    while(!error_sum.compare_exchange_weak(buffer,(buffer - sample_errors[sample_index])))
      buffer = error_sum;

    sample_errors[sample_index] = cost_function->get_feature_error(
      label_samples[sample_index], neuron_data
    );
    
    while(!error_sum.compare_exchange_weak(buffer,(buffer + sample_errors[sample_index])))
      buffer = error_sum;
  }else throw "Sample index out of bounds!";
}

} /* namespace sparse_net_library*/
