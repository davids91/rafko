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

#include "rafko_mainframe/services/rafko_gpu_context.hpp"

#include <stdexcept>

#include "rafko_protocol/solution.pb.h"
#include "rafko_net/services/solution_builder.hpp"
#include "rafko_mainframe/services/rafko_dummies.hpp"
#include "rafko_mainframe/services/rafko_assertion_logger.hpp"

namespace rafko_mainframe{

RafkoGPUContext::RafkoGPUContext(
  cl::Context&& context_, cl::Device device_,
  rafko_mainframe::RafkoSettings settings_, rafko_net::RafkoNet& neural_network_
):RafkoContext(settings_)
, network(neural_network_)
, network_solution(rafko_net::SolutionBuilder(settings).build(network))
, weight_adapter(network, *network_solution, settings)
, agent(rafko_net::SolutionSolver::Builder(*network_solution, settings).build())
, environment(std::make_unique<RafkoDummyEnvironment>(
  network.input_data_size(), network.output_neuron_number())
)
, objective(std::make_unique<RafkoDummyObjective>())
, weight_updater(rafko_gym::UpdaterFactory::build_weight_updater(network, rafko_gym::weight_updater_default, settings))
, neuron_outputs_to_evaluate( /* For every thread, 1 sequence is evaluated.. */
  (settings.get_max_processing_threads() * environment->get_sequence_size() + 1u),
  std::vector<double>(network.output_neuron_number()) /* ..plus for the label errors one additional vector is needed */
)
, execution_threads(settings.get_max_processing_threads())
, opencl_context(context_)
, opencl_device(device_)
, opencl_queue(opencl_context, opencl_device)
, solution_phase( opencl_context, opencl_device, opencl_queue, agent )
, error_phase(
  opencl_context, opencl_device, opencl_queue,
  std::make_shared<RafkoDummyGPUStrategyPhase>(
    RafkoNBufShape({network.output_neuron_number(), network.output_neuron_number()}),
    RafkoNBufShape({1u})
  )
)
{
  neuron_outputs_to_evaluate.back().resize(environment->get_number_of_label_samples());
  upload_weight_table_to_device(); /*!Note: Also sets device_weight_table_size*/
  refresh_objective();
}

void RafkoGPUContext::upload_weight_to_device(std::uint32_t weight_index){
  std::vector<std::pair<std::uint32_t,std::uint32_t>> relevant_partial_weights = weight_adapter.get_relevant_partial_weight_indices_for(
    weight_index
  );
  std::uint32_t weight_table_offset = 0u;
  std::uint32_t partial_index = 0u;
  RFASSERT_LOG("Starting to upload a single weight to device..");
  for(const std::pair<std::uint32_t,std::uint32_t>& index_pair : relevant_partial_weights){
    while(partial_index < std::get<0>(index_pair)){
      weight_table_offset += network_solution->partial_solutions(partial_index).weight_table_size();
      ++partial_index;
    }
    std::uint32_t weight_index_in_partial = std::get<1>(index_pair);
    double weight_value = network_solution->partial_solutions(partial_index).weight_table(weight_index_in_partial);
    RFASSERT_LOG("Weight index in partial[{}]: {}", partial_index, weight_index_in_partial);

    /* Update weight at weight_table_offset + std::get<1>(index_pair) */
    std::uint32_t current_offset = (sizeof(double) * (agent->get_input_shapes()[0][0] + weight_table_offset + weight_index_in_partial) );
    RFASSERT_LOG(
      "buffer byte offset: {} / {}",
      current_offset, (agent->get_input_shapes()[0].get_byte_size<double>())
    );
    cl_int return_value = opencl_queue.enqueueWriteBuffer(
      solution_phase.get_input_buffer(), CL_TRUE/* blocking */,
      current_offset/*offset: mode */, sizeof(double)/*size*/, &weight_value
    );
    RFASSERT( return_value == CL_SUCCESS );
  }
  RFASSERT_LOG("Weight upload complete!");
}

void RafkoGPUContext::set_network_weight(std::uint32_t weight_index, double weight_value){
  RFASSERT_LOG("Setting weight[{}] to {}", weight_index, weight_value);
  RFASSERT( static_cast<std::int32_t>(weight_index) < network.weight_table_size() );
  network.set_weight_table(weight_index, weight_value);
  weight_adapter.update_solution_with_weight(weight_index);
  upload_weight_to_device(weight_index);
}

void RafkoGPUContext::set_network_weights(const std::vector<double>& weights){
  RFASSERT_LOGV(weights, "Setting weights to:");
  RFASSERT( static_cast<std::int32_t>(weights.size()) == network.weight_table_size() );
  *network.mutable_weight_table() = {weights.begin(), weights.end()};
  weight_adapter.update_solution_with_weights();
  upload_weight_table_to_device();
}

void RafkoGPUContext::apply_weight_update(const std::vector<double>& weight_delta){
  RFASSERT_LOGV(weight_delta, "Applying weight update! Delta:");
  RFASSERT( static_cast<std::int32_t>(weight_delta.size()) == network.weight_table_size() );
  if(weight_updater->is_finished())
    weight_updater->start();
  weight_updater->iterate(weight_delta);
  weight_adapter.update_solution_with_weights();
  upload_weight_table_to_device();
}

void RafkoGPUContext::upload_weight_table_to_device(){
  RFASSERT_LOG("Uploading weight table to device..");
  std::vector<double> device_weight_table;
  std::uint32_t overall_number_of_weights = 0u;
  for(const rafko_net::PartialSolution& partial : network_solution->partial_solutions()){
    device_weight_table.insert(
      device_weight_table.end(),
      partial.weight_table().begin(), partial.weight_table().end()
    );
    overall_number_of_weights += partial.weight_table_size();
  }

  RFASSERT_LOGV(device_weight_table, "Weight table being uploaded to device:");
  RFASSERT( device_weight_table.size() == overall_number_of_weights );
  device_weight_table_size = device_weight_table.size();

  cl_int return_value = opencl_queue.enqueueWriteBuffer(
    solution_phase.get_input_buffer(), CL_TRUE/*blocking*/, sizeof(double)/*offset*/,
    (sizeof(double) * device_weight_table.size())/*size*/,
    device_weight_table.data()
  );
  RFASSERT( return_value == CL_SUCCESS );
}

void RafkoGPUContext::refresh_objective(){
  RFASSERT_LOG("Refreshing objective in GPU context..");
  objective->set_gpu_parameters(
    environment->get_number_of_label_samples(),
    environment->get_feature_size()
  );
  error_phase.set_strategy(objective);
}

void RafkoGPUContext::set_objective(std::shared_ptr<rafko_gym::RafkoObjective> objective_){
  RFASSERT_LOG("Setting a new objective in GPU context");
  objective.reset();
  objective = objective_;
  refresh_objective();
  RFASSERT_LOG("Last ran evaluation set to `Not evaluation run`");
  last_ran_evaluation = not_eval_run;
}

void RafkoGPUContext::set_weight_updater(rafko_gym::Weight_updaters updater){
  RFASSERT_LOG("Setting weight updater in GPU context to {}", rafko_gym::Weight_updaters_Name(updater));
  weight_updater.reset();
  weight_updater = rafko_gym::UpdaterFactory::build_weight_updater(network, updater, settings);
}

void RafkoGPUContext::set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment_){
  RFASSERT_LOG("Setting environment in GPU context..");
  RFASSERT_LOG("Environment feature size: {} vs. Network output Neuron number: {}", environment_->get_feature_size(), network.output_neuron_number());
  RFASSERT(environment_->get_feature_size() == network.output_neuron_number());
  RFASSERT_LOG("Environment input size: {} vs. Network input size: {}", environment_->get_input_size(), network.input_data_size());
  RFASSERT(environment_->get_input_size() == network.input_data_size());
  environment.reset();
  environment = environment_;
  std::uint32_t old_output_buffer_num = neuron_outputs_to_evaluate.size();
  std::uint32_t new_output_buffer_num = settings.get_max_processing_threads() * environment->get_sequence_size() + 1u;
  neuron_outputs_to_evaluate.resize(new_output_buffer_num);
  if(old_output_buffer_num < new_output_buffer_num){
    for(std::uint32_t buffer_index = old_output_buffer_num-1; buffer_index < new_output_buffer_num; ++buffer_index){
      neuron_outputs_to_evaluate[buffer_index].resize(environment->get_feature_size());
    }
  }
  neuron_outputs_to_evaluate.back().resize(environment->get_number_of_label_samples());
  RFASSERT_LOG(
    "Agent sequence parameters: {} sequences; {} sequence_size; {} prefill inputs",
    environment->get_number_of_sequences(),
    environment->get_sequence_size(),
    environment->get_prefill_inputs_number()
  );
  agent->set_sequence_params(
    environment->get_number_of_sequences(),
    environment->get_sequence_size(),
    environment->get_prefill_inputs_number()
  );
  solution_phase.set_strategy(agent);
  upload_weight_table_to_device();
  refresh_objective();
  RFASSERT_LOG("Last ran evaluation set to `Not evaluation run`");
  last_ran_evaluation = not_eval_run;
}

std::vector<cl::Event> RafkoGPUContext::upload_agent_output(
  std::uint32_t sequences_to_upload, std::uint32_t start_index_inside_sequence, std::uint32_t sequence_truncation
){
  [[maybe_unused]]cl_int return_value;
  RFASSERT_LOG(
    "Uploading agent outputs:  sequences to upload: {}; start index inside sequence: {}; sequence truncation: {}",
    sequences_to_upload, start_index_inside_sequence, sequence_truncation
  );

  std::uint32_t elements_to_upload = (sequences_to_upload * sequence_truncation);
  /*!Note: elements == features */
  std::uint32_t byte_offset_solution_phase = 0u;
  std::uint32_t byte_offset_error_phase = 0u;

  RFASSERT( (start_index_inside_sequence + sequence_truncation) <= environment->get_sequence_size() );
  RFASSERT( 0u < sequence_truncation );

  const std::uint32_t truncation_starts_in_sequence = (environment->get_prefill_inputs_number() + start_index_inside_sequence);
  std::vector<cl::Event> events(elements_to_upload);
  for(std::uint32_t sequence_index = 0; sequence_index < sequences_to_upload; ++sequence_index){
    for(std::uint32_t label_index = 0; label_index < (environment->get_sequence_size() + environment->get_prefill_inputs_number()); ++label_index){
      /* add neuron_array_offset */
      byte_offset_solution_phase += ( (network.neuron_array_size() - network.output_neuron_number()) * sizeof(double) );
      if((truncation_starts_in_sequence <= label_index)&&(label_index < (truncation_starts_in_sequence + sequence_truncation))){
        RFASSERT_LOG(
          "copying from agent[{}] to objective[{} + {}] / {} bytes..",
          byte_offset_solution_phase, byte_offset_error_phase,
          (network.output_neuron_number() * sizeof(double)), objective->get_input_shapes()[0].get_byte_size<double>()
        );
        return_value = opencl_queue.enqueueCopyBuffer( /* Upload sequence */
          solution_phase.get_output_buffer() /*src*/, error_phase.get_input_buffer() /*dst*/,
          byte_offset_solution_phase/*src_offset*/, byte_offset_error_phase/*dst_offset*/,
          (network.output_neuron_number() * sizeof(double))/*size*/,
          NULL /*events to wait for*/, &events[(sequence_index * sequence_truncation) + label_index - truncation_starts_in_sequence]
        );
        RFASSERT( return_value == CL_SUCCESS );

        byte_offset_error_phase += (network.output_neuron_number() * sizeof(double));
      }
      byte_offset_solution_phase += (network.output_neuron_number() * sizeof(double));
    }
  }
  return events;
}

double RafkoGPUContext::error_post_process(double raw_error, std::uint32_t labels_evaluated){
  double error_value = raw_error;
  double divisor = std::max(labels_evaluated, 1u);
  double performance_error = solution_phase.acquire_output(
    1u, agent->get_output_shapes()[0][0] /* first output, after the size of the first output */
  )[0];
  RFASSERT_LOG(
    "Error post process: raw error value: {}; performance error: {}; divisor: {}",
    error_value, performance_error, divisor
  );
  return ( (error_value + performance_error) / divisor );
}

double RafkoGPUContext::full_evaluation(){
  [[maybe_unused]]cl_int return_value;
  std::vector<cl::Event> label_events;
  RFASSERT_SCOPE(GPU_FULL_EVALUATION);

  if(last_ran_evaluation != full_eval_run){
    /* upload mode info */
    cl::Event fill_event;
    return_value = opencl_queue.enqueueFillBuffer<double>(
      solution_phase.get_input_buffer(), (0.0)/*the double value*/,
      0u /*offset*/, sizeof(double)/*size(bytes)*/,
      NULL/*events to wit for*/, &fill_event
    );
    RFASSERT( return_value == CL_SUCCESS );
    return_value = fill_event.wait();
    RFASSERT( return_value == CL_SUCCESS );

    std::vector<cl::Event> input_events = environment->upload_inputs_to_buffer(
      opencl_queue, solution_phase.get_input_buffer(),
      sizeof(double) * (device_weight_table_size + agent->get_input_shapes()[0][0])/*buffer_start_byte_offset*/,
      0u/*sequence_start_index*/, 0u/*buffer_sequence_start_index*/,
      environment->get_number_of_sequences()/*sequences_to_upload*/
    );

    label_events = environment->upload_labels_to_buffer(
      opencl_queue, error_phase.get_input_buffer(), (
        environment->get_number_of_label_samples()
        * environment->get_feature_size() * sizeof(double)
      ) /*buffer_start_byte_offset*/,
      0u/*sequence_start_index*/, 0/*buffer_sequence_start_index*/,
      environment->get_number_of_sequences()/*sequences_to_upload*/,
      0u/*start_index_inside_sequence*/, environment->get_sequence_size()/*sequence_truncation*/
    );

    for(cl::Event& input_event : input_events){
      return_value = input_event.wait();
      RFASSERT( return_value == CL_SUCCESS );
    }
  }

  /* run feature phase */
  cl::EnqueueArgs enque_arguments = std::make_from_tuple<cl::EnqueueArgs>(
    std::tuple_cat(std::tie(opencl_queue), agent->get_solution_space())
  );
  solution_phase( enque_arguments );

  /* upload agent output into error phase inputs */
  std::vector<cl::Event> features_events = upload_agent_output(
    environment->get_number_of_sequences(),
    0u/*start_index_inside_sequence*/, environment->get_sequence_size()/*sequence_truncation*/
  );

  for(cl::Event& features_event : features_events){
    return_value = features_event.wait();
    RFASSERT( return_value == CL_SUCCESS );
  }

  for(cl::Event& label_event : label_events){
    return_value = label_event.wait();
    RFASSERT( return_value == CL_SUCCESS );
  }

  /* run error phase */
  cl::EnqueueArgs error_enque_arguments = std::make_from_tuple<cl::EnqueueArgs>(
    std::tuple_cat(std::tie(opencl_queue), objective->get_solution_space())
  );
  error_phase( error_enque_arguments );

  RFASSERT_LOG("Last ran evaluation set to `Full evaluation run`");
  last_ran_evaluation = full_eval_run;
  return -error_post_process(error_phase.acquire_output(1u)[0], environment->get_number_of_label_samples());
}

double RafkoGPUContext::stochastic_evaluation(bool to_seed, std::uint32_t seed_value){
  [[maybe_unused]]cl_int return_value;
  std::vector<cl::Event> input_events;
  std::vector<cl::Event> label_events;
  RFASSERT_SCOPE(GPU_STOCHASTIC_EVALUATION);

  if(to_seed){
    srand(seed_value);
    RFASSERT_LOG("Seeded run: last used seed: {}; current seed: {}", last_used_seed, seed_value);
  }
  const std::uint32_t used_minibatch_size = std::min(settings.get_minibatch_size(), environment->get_number_of_sequences());
  const std::uint32_t used_sequence_truncation = std::min( settings.get_memory_truncation(), environment->get_sequence_size() );
  const std::uint32_t start_index_inside_sequence = ( rand()%(environment->get_sequence_size() - used_sequence_truncation + 1) );
  RFASSERT_LOG(
    "Used minibatch size: {}; sequence_truncation: {}; start index inside sequence: {}",
    used_minibatch_size, used_sequence_truncation, start_index_inside_sequence
  );
  if(
    (last_ran_evaluation != random_eval_run)
    ||(last_used_seed != seed_value)
    ||(!last_random_eval_was_seeded)
  ){
    cl::Event fill_event;
    RFASSERT_LOG("Updating evaluation buffer..");
    return_value = opencl_queue.enqueueFillBuffer<double>( /* upload mode info */
      solution_phase.get_input_buffer(), (0)/*the double value*/,
      0u /*offset*/, sizeof(double)/*size(bytes)*/,
      NULL/*events to wit for*/, &fill_event
    );
    RFASSERT( return_value == CL_SUCCESS );
    return_value = fill_event.wait();
    RFASSERT( return_value == CL_SUCCESS );

    /* upload pseudo-random labels and inputs */
    std::uint32_t uploaded_sequences = 0u;
    while(uploaded_sequences < used_minibatch_size){
      std::uint32_t sequences_to_upload = rand()%(used_minibatch_size - uploaded_sequences + 1u);
      std::uint32_t sequence_start_index = rand()%(environment->get_number_of_sequences() - sequences_to_upload + 1u);
      RFASSERT_LOG("Uploading {} sequences starting from {}", sequences_to_upload, sequence_start_index);
      std::vector<cl::Event> input_events_ = environment->upload_inputs_to_buffer(
        opencl_queue, solution_phase.get_input_buffer(),
        sizeof(double) * (device_weight_table_size + agent->get_input_shapes()[0][0])/*buffer_start_byte_offset*/,
        sequence_start_index, uploaded_sequences/*buffer_sequence_start_index*/,
        sequences_to_upload/*sequences_to_upload*/
      );
      input_events.insert(
        input_events.end(),
        input_events_.begin(), input_events_.end()
      );
      std::vector<cl::Event> label_events_ = environment->upload_labels_to_buffer(
        opencl_queue, error_phase.get_input_buffer(), (
          environment->get_number_of_label_samples() * environment->get_feature_size() * sizeof(double)
        ) /*buffer_start_byte_offset*/,
        sequence_start_index, uploaded_sequences/*buffer_sequence_start_index*/,
        sequences_to_upload/*sequences_to_upload*/,
        start_index_inside_sequence, used_sequence_truncation
      );
      label_events.insert(
        label_events.end(),
        label_events_.begin(), label_events_.end()
      );
      uploaded_sequences += sequences_to_upload;
    }/*while(uploaded_sequences < used_minibatch_size)*/
  }

  if(to_seed){
    last_used_seed = seed_value;
    last_random_eval_was_seeded = true;
  }

  for(cl::Event& event : input_events){
    return_value = event.wait();
    RFASSERT( return_value == CL_SUCCESS );
  }

  /* run feature phase */
  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> sol_space = agent->get_solution_space();
  std::get<1>(sol_space) = cl::NDRange(used_minibatch_size);
  cl::EnqueueArgs enque_arguments = std::make_from_tuple<cl::EnqueueArgs>(
    std::tuple_cat(std::tie(opencl_queue), sol_space)
  );
  solution_phase( enque_arguments );

  /* upload agent output into error phase inputs */
  std::vector<cl::Event> features_events = upload_agent_output(
    used_minibatch_size, start_index_inside_sequence, used_sequence_truncation
  );

  /* fill the rest of the output buffer with the label value */
  cl::Event fill_event;
  std::uint32_t uploaded_bytes_count = (environment->get_feature_size() * used_sequence_truncation * used_minibatch_size * sizeof(double));
  std::uint32_t minibatch_labels_byte_size = (environment->get_feature_size() * environment->get_sequence_size() * used_minibatch_size * sizeof(double));
  RFASSERT_LOG(
    "Copying label values[{}] to agent output buffer[{} + {}] as dummy data..",
    minibatch_labels_byte_size, uploaded_bytes_count, (minibatch_labels_byte_size - uploaded_bytes_count)
  );

  return_value = opencl_queue.enqueueCopyBuffer(
    error_phase.get_input_buffer()/*src*/,  error_phase.get_input_buffer()/*dst*/,
    minibatch_labels_byte_size/*src_offset*/, uploaded_bytes_count/*dst_offset*/,
    (minibatch_labels_byte_size - uploaded_bytes_count)/*size*/,
    NULL /*events to wait for*/, &fill_event
  );
  RFASSERT( return_value == CL_SUCCESS );

  for(cl::Event& features_event : features_events){
    return_value = features_event.wait();
    RFASSERT( return_value == CL_SUCCESS );
  }
  for(cl::Event& event : label_events){
    return_value = event.wait();
    RFASSERT( return_value == CL_SUCCESS );
  }

  return_value = fill_event.wait();
  RFASSERT( return_value == CL_SUCCESS );

  /* run error phase */
  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> error_sol_space = objective->get_solution_space();
  std::get<1>(error_sol_space) = cl::NDRange(used_minibatch_size * used_sequence_truncation);
  cl::EnqueueArgs error_enque_arguments = std::make_from_tuple<cl::EnqueueArgs>(
    std::tuple_cat(std::tie(opencl_queue), error_sol_space)
  );
  error_phase( error_enque_arguments );
  RFASSERT_LOG("Last ran evaluation set to `Random evaluation run`");
  last_ran_evaluation = random_eval_run;
  return -error_post_process(
    error_phase.acquire_output(1u)[0], (used_minibatch_size * environment->get_sequence_size())
  );
}

rafko_utilities::ConstVectorSubrange<> RafkoGPUContext::solve(
  const std::vector<double>& input,
  bool reset_neuron_data, std::uint32_t thread_index
){
  RFASSERT_SCOPE(GPU_STANDALONE_SOLVE);
  RFASSERT_LOG("Thread index in solve: {}", thread_index);
  RFASSERT(0 == thread_index);
  if(0u != thread_index)
    throw std::runtime_error("Multi-threaded openCL Environment not supported!");

  [[maybe_unused]]cl_int return_value;
  cl::Event fill_event;
  const std::uint32_t network_memory_slots = std::max(2u, (network_solution->network_memory_length() - 1u));
  const std::size_t network_used_bytes = sizeof(double) * network_memory_slots * network_solution->neuron_number();

  if(reset_neuron_data || (last_ran_evaluation != not_eval_run) ){
    RFASSERT_LOG("Not resetting agent data..");
    return_value = opencl_queue.enqueueFillBuffer<double>(
      solution_phase.get_output_buffer(), (0.0)/* the data(pattern) value */,
      0u/*offset*/, network_used_bytes/*size*/,
      NULL/*events to wait for*/, &fill_event
    );
    RFASSERT( return_value == CL_SUCCESS );

    return_value = fill_event.wait();
    RFASSERT( return_value == CL_SUCCESS );
  }else{ /* Neuron memory not resetted, keep network memory consistent */
    const std::uint32_t network_memory_span_bytes = network_solution->neuron_number() * sizeof(double);
    RFASSERT_LOG(
      "Resetting agent data; Memory slots: {}; Memory size in bytes: {}",
      network_memory_slots, network_memory_span_bytes
    );
    for(std::uint32_t memory_slot = 0; memory_slot < (network_memory_slots-1u); ++memory_slot){
      return_value = opencl_queue.enqueueCopyBuffer(
        solution_phase.get_output_buffer()/*src*/, solution_phase.get_output_buffer() /*dst*/,
        ((memory_slot + 1u) * network_memory_span_bytes)/*src_offset*/,
        (memory_slot * network_memory_span_bytes)/*dst_offset*/,
        network_memory_span_bytes/*size*/, NULL, &fill_event
      );
      RFASSERT( return_value == CL_SUCCESS );
      return_value = fill_event.wait();
      RFASSERT( return_value == CL_SUCCESS );
    }
  }

  /* upload mode info */
  return_value = opencl_queue.enqueueFillBuffer<double>(
    solution_phase.get_input_buffer(), (69.420)/*the double value*/,
    0u /*offset*/, sizeof(double)/*size(bytes)*/,
    NULL/*events to wit for*/, &fill_event
  );
  RFASSERT( return_value == CL_SUCCESS );
  return_value = fill_event.wait();
  RFASSERT( return_value == CL_SUCCESS );

  /* upload inputs */
  return_value = opencl_queue.enqueueWriteBuffer(
    solution_phase.get_input_buffer(), CL_TRUE,
    (sizeof(double) * (device_weight_table_size + 1u))/*offset: mode and weights*/,
    (sizeof(double) * input.size())/*size*/,
    input.data(), NULL
  );
  RFASSERT( return_value == CL_SUCCESS );

  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> sol_space = agent->get_solution_space();
  std::get<1>(sol_space) = cl::NDRange(1);
  cl::EnqueueArgs enq = std::make_from_tuple<cl::EnqueueArgs>( std::tuple_cat(std::tie(opencl_queue), sol_space) );
  solution_phase( enq );

  std::uint32_t output_array_start = ( /* the end of the last memory slot contains the network data */
    (std::max(2u, network_solution->network_memory_length()) * network.neuron_array_size()) - network.output_neuron_number()
  );
  RFASSERT_LOG("Output array start: {}", output_array_start);
  if(static_cast<std::int32_t>(standalone_solution_result.size()) == network.neuron_array_size()){
    RFASSERT_LOG("Loading output to already allocated vector..");
    solution_phase.load_output(standalone_solution_result.data(), network.output_neuron_number(), output_array_start);
  }else{
    RFASSERT_LOG("Acquiring output and moving it to new vector..");
    std::unique_ptr<double[]> output_ptr = solution_phase.acquire_output(network.output_neuron_number(), output_array_start);
    standalone_solution_result = std::vector<double>(output_ptr.get(), output_ptr.get() + network.output_neuron_number());
  }
  RFASSERT_LOGV(standalone_solution_result, "Resulting output:");

  RFASSERT_LOG("Last ran evaluation set to `Not evaluation run`");
  last_ran_evaluation = not_eval_run;

  return { standalone_solution_result.end() - network.output_neuron_number(), standalone_solution_result.end() };
}

} /* namespace rafko_mainframe */
