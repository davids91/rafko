/*! This file is part of davids91/Rafko.
 *
 *    Rafko is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    Rafko is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with Rafko.  If not, see <https://www.gnu.org/licenses/> or
 *    <https://github.com/davids91/rafko/blob/master/LICENSE>
 */

#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/solution.pb.h"
#include "rafko_net/services/rafko_network_feature.h"
#include "rafko_net/services/synapse_iterator.h"
#include "rafko_net/services/rafko_net_builder.h"
#include "rafko_gym/models/rafko_cost.h"
#include "rafko_gym/models/rafko_dataset_wrapper.h"
#include "rafko_mainframe/services/rafko_cpu_context.h"
#if(RAFKO_USES_OPENCL)
#include "rafko_mainframe/services/rafko_gpu_context.h"
#endif/*(RAFKO_USES_OPENCL)*/


#include "test/test_utility.h"

namespace rafko_net_test {

TEST_CASE("Test if L1 regularization calculates the expected error", "[L1][regularization][features]"){
  google::protobuf::Arena arena;
  uint32 sequence_size = 6u;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(uint32 variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNetBuilder builder = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range(double_literal(1.0));

    std::set<uint32> affected_layers;
    for(uint32 try_add = 0; try_add < 10; ++try_add){
      uint32 layer = (rand()%6);
      affected_layers.insert(layer);
      builder.add_feature_to_layer(layer, rafko_net::neuron_group_feature_l1_regularization);
    }
    std::vector<uint32> layer_sizes = {
      static_cast<uint32>(rand()%3) + 1u,
      static_cast<uint32>(rand()%3) + 1u,
      static_cast<uint32>(rand()%3) + 1u,
      static_cast<uint32>(rand()%3) + 1u,
      static_cast<uint32>(rand()%3) + 1u,
      2
    };
    rafko_net::RafkoNet* network = builder.dense_layers(layer_sizes);

    /* store which Neuron index belongs to which layer index */
    std::map<uint32,uint32> layer_index_values;
    uint32 layer_index_value = 0u;
    uint32 layer_start = 0u;
    for(const uint32& layer_size : layer_sizes){
      for(uint32 neuron_index = layer_start; neuron_index < (layer_start + layer_size); ++neuron_index){
        layer_index_values.insert({neuron_index,layer_index_value});
        /*!Note: Neuron index is the key */
      }
      layer_start += layer_size;
      ++layer_index_value;
    }

    /* calculate l1 errors manually and check for the actual values */
    std::map<uint32,sdouble32> layer_errors;
    sdouble32 sum_error;
    for(const uint32& layer_index : affected_layers){
      layer_start = std::accumulate(layer_sizes.begin(), layer_sizes.begin() + layer_index, double_literal(0.0));
      sdouble32 layer_error = double_literal(0.0);
      for(uint32 neuron_index = layer_start; neuron_index < (layer_start + layer_sizes[layer_index]); ++neuron_index){
        rafko_net::SynapseIterator<>::iterate(network->neuron_array(neuron_index).input_weights(),
        [&network, &layer_error](uint32 weight_index){
          layer_error += std::abs(network->weight_table(weight_index));
        });
      }/*for(neurons in the affected layer)*/
      sum_error += layer_error;
      layer_errors.insert({layer_index, layer_error});
    }/*for(the affected layers)*/

    /* declare an executor */
    std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>> exec_threads;
    exec_threads.push_back(std::make_unique<rafko_utilities::ThreadGroup>(settings.get_max_processing_threads()));
    rafko_net::RafkoNetworkFeature features(exec_threads);

    for(const rafko_net::FeatureGroup& group : network->neuron_group_features()){
      if(group.feature() == rafko_net::neuron_group_feature_l1_regularization){
        sdouble32 reference_error = layer_errors[layer_index_values[group.relevant_neurons(0).starts()]];
        CHECK(
          Catch::Approx(reference_error).epsilon(double_literal(0.00000000000001))
          == features.calculate_performance_relevant(group, *network)
        );
      }/*if(feature is l1 regularization)*/
    }/*for(all feature groups in network)*/
  }/*for(50 variants)*/
}

TEST_CASE("Test if L2 regularization calculates the expected error", "[L2][regularization][features]"){
  google::protobuf::Arena arena;
  uint32 sequence_size = 6u;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(uint32 variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNetBuilder builder = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range(double_literal(1.0));

    std::set<uint32> affected_layers;
    for(uint32 try_add = 0; try_add < 10; ++try_add){
      uint32 layer = (rand()%6);
      affected_layers.insert(layer);
      builder.add_feature_to_layer(layer, rafko_net::neuron_group_feature_l2_regularization);
    }
    std::vector<uint32> layer_sizes = {
      static_cast<uint32>(rand()%3) + 1u,
      static_cast<uint32>(rand()%3) + 1u,
      static_cast<uint32>(rand()%3) + 1u,
      static_cast<uint32>(rand()%3) + 1u,
      static_cast<uint32>(rand()%3) + 1u,
      2
    };
    rafko_net::RafkoNet* network = builder.dense_layers(layer_sizes);

    /* store which Neuron index belongs to which layer index */
    std::map<uint32,uint32> layer_index_values;
    uint32 layer_index_value = 0u;
    uint32 layer_start = 0u;
    for(const uint32& layer_size : layer_sizes){
      for(uint32 neuron_index = layer_start; neuron_index < (layer_start + layer_size); ++neuron_index){
        layer_index_values.insert({neuron_index,layer_index_value});
        /*!Note: Neuron index is the key */
      }
      layer_start += layer_size;
      ++layer_index_value;
    }

    /* calculate l1 errors manually and check for the actual values */
    std::map<uint32,sdouble32> layer_errors;
    sdouble32 sum_error;
    for(const uint32& layer_index : affected_layers){
      layer_start = std::accumulate(layer_sizes.begin(), layer_sizes.begin() + layer_index, double_literal(0.0));
      sdouble32 layer_error = double_literal(0.0);
      for(uint32 neuron_index = layer_start; neuron_index < (layer_start + layer_sizes[layer_index]); ++neuron_index){
        rafko_net::SynapseIterator<>::iterate(network->neuron_array(neuron_index).input_weights(),
        [&network, &layer_error](uint32 weight_index){
          layer_error += std::pow(network->weight_table(weight_index),double_literal(2.0));
        });
      }/*for(neurons in the affected layer)*/
      sum_error += layer_error;
      layer_errors.insert({layer_index, layer_error});
    }/*for(the affected layers)*/

    /* create a feature executor */
    std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>> exec_threads;
    exec_threads.push_back(std::make_unique<rafko_utilities::ThreadGroup>(settings.get_max_processing_threads()));
    rafko_net::RafkoNetworkFeature features(exec_threads);

    for(const rafko_net::FeatureGroup& group : network->neuron_group_features()){
      if(group.feature() == rafko_net::neuron_group_feature_l2_regularization){
        sdouble32 reference_error = layer_errors[layer_index_values[group.relevant_neurons(0).starts()]];
        CHECK(
          Catch::Approx(reference_error).epsilon(double_literal(0.00000000000001))
          == features.calculate_performance_relevant(group, *network)
        );
      }/*if(feature is l1 regularization)*/
    }/*for(all feature groups in network)*/
  }/*for(10 variants)*/
}

TEST_CASE("Test if L1 and L2 regularization errors are added correctly to CPU context", "[CPU][context][regularization][features]"){
  google::protobuf::Arena arena;
  uint32 feature_size = 2u;
  uint32 sequence_size = 6u;
  uint32 number_of_sequences = rand()%10 + 1;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(uint32 variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNetBuilder builder = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range(double_literal(1.0));

    std::set<uint32> affected_layers;
    for(uint32 try_add = 0; try_add < 10; ++try_add){
      uint32 layer = (rand()%6);
      affected_layers.insert(layer);
      builder.add_feature_to_layer(layer, rafko_net::neuron_group_feature_l2_regularization);
    }
    std::vector<uint32> layer_sizes = {
      static_cast<uint32>(rand()%3) + 1u,
      static_cast<uint32>(rand()%3) + 1u,
      static_cast<uint32>(rand()%3) + 1u,
      static_cast<uint32>(rand()%3) + 1u,
      static_cast<uint32>(rand()%3) + 1u,
      feature_size
    };
    rafko_net::RafkoNet* network = builder.dense_layers(layer_sizes);
    rafko_net::RafkoNet unregulated_network = rafko_net::RafkoNet(*network);

    /* declare an executor */
    std::vector<std::unique_ptr<rafko_utilities::ThreadGroup>> exec_threads;
    exec_threads.push_back(std::make_unique<rafko_utilities::ThreadGroup>(settings.get_max_processing_threads()));
    rafko_net::RafkoNetworkFeature features(exec_threads);

    /* Remove weight regularization from a copy network, and calculate the error difference */
    unregulated_network.mutable_neuron_group_features()->Clear();
    sdouble32 error_difference = double_literal(0.0);
    for(const rafko_net::FeatureGroup& feature : network->neuron_group_features()){
      if( /* Add back all the irrelevant features */
        (feature.feature() != rafko_net::neuron_group_feature_l1_regularization)
        &&(feature.feature() != rafko_net::neuron_group_feature_l2_regularization)
      ){
        *unregulated_network.add_neuron_group_features() = feature;
      }else{
        error_difference += features.calculate_performance_relevant(feature, unregulated_network);
      }
    }

    /* Create CPU contexts and an environment */
    rafko_mainframe::RafkoCPUContext regulated_context(*network, settings);
    rafko_mainframe::RafkoCPUContext unregulated_context(unregulated_network, settings);
    std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
      settings, rafko_gym::cost_function_squared_error
    );
    std::unique_ptr<rafko_gym::DataSet> dataset( rafko_test::create_dataset(
      2/* input size */, feature_size,
      number_of_sequences, sequence_size, 2/*prefill_size*/,
      rand()%100/*expected_label*/, double_literal(1.0)
    ) );
    std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(*dataset);

    REQUIRE( /* because the evaluation provides fittness value, the error difference needs to be substracted..*/
      Catch::Approx(regulated_context.full_evaluation()).epsilon(double_literal(0.00000000000001))
      == (unregulated_context.full_evaluation() - error_difference)
    );

    regulated_context.set_environment(environment);
    unregulated_context.set_environment(environment);
    regulated_context.set_objective(objective);
    unregulated_context.set_objective(objective);

    REQUIRE(
      Catch::Approx(regulated_context.full_evaluation()).epsilon(double_literal(0.00000000000001))
      == (unregulated_context.full_evaluation() - error_difference)
    );
  }/*for(10 variants)*/
}

#if(RAFKO_USES_OPENCL)
TEST_CASE("Test if L1 and L2 regularization errors are added correctly to GPU context", "[GPU][context][regularization][features]"){
  google::protobuf::Arena arena;
  uint32 feature_size = 2u;
  uint32 sequence_size = 6u;
  uint32 number_of_sequences = rand()%10 + 1;
  rafko_mainframe::RafkoSettings settings = rafko_mainframe::RafkoSettings()
    .set_max_processing_threads(4).set_memory_truncation(sequence_size)
    .set_arena_ptr(&arena)
    .set_minibatch_size(10);
  for(uint32 variant = 0u; variant < 10u; ++variant){
    rafko_net::RafkoNetBuilder builder = rafko_net::RafkoNetBuilder(settings)
      .input_size(2).expected_input_range(double_literal(1.0));

    std::vector<uint32> layer_sizes = {
      static_cast<uint32>(rand()%3) + 1u,
      // static_cast<uint32>(rand()%3) + 1u,
      // static_cast<uint32>(rand()%3) + 1u,
      // static_cast<uint32>(rand()%3) + 1u,
      static_cast<uint32>(rand()%3) + 1u,
      feature_size
    };
    for(uint32 try_add = 0; try_add < 10; ++try_add){
      builder.add_feature_to_layer( (rand()%layer_sizes.size()), rafko_net::neuron_group_feature_l1_regularization );
      /*TODO: Add L2 regularization as well */
    }

    rafko_net::RafkoNet* network = builder.dense_layers(layer_sizes);
    rafko_net::RafkoNet network_copy = rafko_net::RafkoNet(*network);

    /* Create environments */
    std::shared_ptr<rafko_gym::RafkoObjective> objective = std::make_shared<rafko_gym::RafkoCost>(
      settings, rafko_gym::cost_function_squared_error
    );
    std::unique_ptr<rafko_gym::DataSet> dataset( rafko_test::create_dataset(
      2/* input size */, feature_size,
      number_of_sequences, sequence_size, 2/*prefill_size*/,
      rand()%100/*expected_label*/, double_literal(1.0)/*label_delta_per_feature*/
    ) );
    std::shared_ptr<rafko_gym::RafkoDatasetWrapper> environment = std::make_shared<rafko_gym::RafkoDatasetWrapper>(*dataset);

    /* create GPU and CPU contexts */
    rafko_mainframe::RafkoCPUContext cpu_context(network_copy, settings);
    std::unique_ptr<rafko_mainframe::RafkoGPUContext> gpu_context;
    REQUIRE_NOTHROW(
      gpu_context = (
        rafko_mainframe::RafkoGPUContext::Builder(*network, settings)
          .select_platform().select_device()
          .build()
      )
    );

    REQUIRE(
      Catch::Approx(cpu_context.full_evaluation()).epsilon(double_literal(0.00000000000001))
      == gpu_context->full_evaluation()
    );

    cpu_context.set_environment(environment);
    gpu_context->set_environment(environment);
    cpu_context.set_objective(objective);
    gpu_context->set_objective(objective);

    REQUIRE(
      Catch::Approx(cpu_context.full_evaluation()).epsilon(double_literal(0.00000000000001))
      == gpu_context->full_evaluation()
    );
  }/*for(10 variants)*/
}
#endif/*(RAFKO_USES_OPENCL)*/

} /* namespace rafko_net_test */
