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
  cl::Context&& context, cl::Device device,
  rafko_net::RafkoNet& neural_network, std::shared_ptr<rafko_mainframe::RafkoSettings> settings,
  std::shared_ptr<rafko_gym::RafkoObjective> objective
):RafkoContext(settings)
, m_network(neural_network)
, m_networkSolution(rafko_net::SolutionBuilder(*m_settings).build(m_network))
, m_weightAdapter(m_network, *m_networkSolution, *m_settings)
, m_agent(rafko_net::SolutionSolver::Builder(m_networkSolution, *m_settings).build())
, m_environment(std::make_unique<RafkoDummyEnvironment>(
  m_network.input_data_size(), m_network.output_neuron_number())
)
, m_objective(objective)
, m_weightUpdater(rafko_gym::UpdaterFactory::build_weight_updater(m_network, rafko_gym::weight_updater_default, *m_settings))
, m_neuronOutputsToEvaluate( /* For every thread, 1 sequence is evaluated.. */
  (m_settings->get_max_processing_threads() * m_environment->get_sequence_size() + 1u),
  std::vector<double>(m_network.output_neuron_number()) /* ..plus for the label errors one additional vector is needed */
)
, m_executionThreads(m_settings->get_max_processing_threads())
, m_openclContext(context)
, m_openclDevice(device)
, m_openclQueue(m_openclContext, m_openclDevice)
, m_solutionPhase(m_openclContext, m_openclDevice, m_openclQueue, m_agent)
, m_errorPhase(
  m_openclContext, m_openclDevice, m_openclQueue,
  std::make_shared<RafkoDummyGPUStrategyPhase>(
    RafkoNBufShape({m_network.output_neuron_number(), m_network.output_neuron_number()}),
    RafkoNBufShape({1u})
  )
)
{
  m_neuronOutputsToEvaluate.back().resize(m_environment->get_number_of_label_samples());
  upload_weight_table_to_device(); /*!Note: Also sets device_weight_table_size*/
  refresh_objective();
}

void RafkoGPUContext::upload_weight_to_device(std::uint32_t weight_index){
  std::vector<std::pair<std::uint32_t,std::uint32_t>> relevant_partial_weights = m_weightAdapter.get_relevant_partial_weight_indices_for(
    weight_index
  );
  std::uint32_t weight_table_offset = 0u;
  std::uint32_t partial_index = 0u;
  RFASSERT_LOG("Starting to upload a single weight to device..");
  for(const std::pair<std::uint32_t,std::uint32_t>& index_pair : relevant_partial_weights){
    while(partial_index < std::get<0>(index_pair)){
      weight_table_offset += m_networkSolution->partial_solutions(partial_index).weight_table_size();
      ++partial_index;
    }
    std::uint32_t weight_index_in_partial = std::get<1>(index_pair);
    double weight_value = m_networkSolution->partial_solutions(partial_index).weight_table(weight_index_in_partial);
    RFASSERT_LOG("Weight index in partial[{}]: {}", partial_index, weight_index_in_partial);

    /* Update weight at weight_table_offset + std::get<1>(index_pair) */
    std::uint32_t current_offset = (sizeof(double) * (m_agent->get_input_shapes()[0][0] + weight_table_offset + weight_index_in_partial) );
    RFASSERT_LOG(
      "buffer byte offset: {} / {}",
      current_offset, (m_agent->get_input_shapes()[0].get_byte_size<double>())
    );
    cl_int return_value = m_openclQueue.enqueueWriteBuffer(
      m_solutionPhase.get_input_buffer(), CL_TRUE/* blocking */,
      current_offset/*offset: mode */, sizeof(double)/*size*/, &weight_value
    );
    RFASSERT( return_value == CL_SUCCESS );
  }
  RFASSERT_LOG("Weight upload complete!");
}

void RafkoGPUContext::set_network_weight(std::uint32_t weight_index, double weight_value){
  RFASSERT_LOG("Setting weight[{}] to {}", weight_index, weight_value);
  RFASSERT( static_cast<std::int32_t>(weight_index) < m_network.weight_table_size() );
  m_network.set_weight_table(weight_index, weight_value);
  m_weightAdapter.update_solution_with_weight(weight_index);
  upload_weight_to_device(weight_index);
}

void RafkoGPUContext::set_network_weights(const std::vector<double>& weights){
  RFASSERT_LOGV(weights, "Setting weights to:");
  RFASSERT( static_cast<std::int32_t>(weights.size()) == m_network.weight_table_size() );
  *m_network.mutable_weight_table() = {weights.begin(), weights.end()};
  m_weightAdapter.update_solution_with_weights();
  upload_weight_table_to_device();
}

void RafkoGPUContext::apply_weight_update(const std::vector<double>& weight_delta){
  RFASSERT_LOGV(weight_delta, "Applying weight update! Delta:");
  RFASSERT( static_cast<std::int32_t>(weight_delta.size()) == m_network.weight_table_size() );
  if(m_weightUpdater->is_finished())
    m_weightUpdater->start();
  m_weightUpdater->iterate(weight_delta);
  m_weightAdapter.update_solution_with_weights();
  upload_weight_table_to_device();
}

void RafkoGPUContext::upload_weight_table_to_device(){
  RFASSERT_LOG("Uploading weight table to device..");
  std::vector<double> device_weight_table;
  std::uint32_t overall_number_of_weights = 0u;
  for(const rafko_net::PartialSolution& partial : m_networkSolution->partial_solutions()){
    device_weight_table.insert(
      device_weight_table.end(),
      partial.weight_table().begin(), partial.weight_table().end()
    );
    overall_number_of_weights += partial.weight_table_size();
  }

  RFASSERT_LOGV(device_weight_table, "Weight table being uploaded to device:");
  RFASSERT( device_weight_table.size() == overall_number_of_weights );
  m_deviceWeightTableSize = device_weight_table.size();

  cl_int return_value = m_openclQueue.enqueueWriteBuffer(
    m_solutionPhase.get_input_buffer(), CL_TRUE/*blocking*/, sizeof(double)/*offset*/,
    (sizeof(double) * device_weight_table.size())/*size*/,
    device_weight_table.data()
  );
  RFASSERT( return_value == CL_SUCCESS );
}

void RafkoGPUContext::refresh_objective(){
  RFASSERT_LOG("Refreshing objective in GPU context..");
  RFASSERT(static_cast<bool>(m_objective));
  m_objective->set_gpu_parameters(
    m_environment->get_number_of_label_samples(),
    m_environment->get_feature_size()
  );
  m_errorPhase.set_strategy(m_objective);
}

void RafkoGPUContext::set_objective(std::shared_ptr<rafko_gym::RafkoObjective> objective){
  RFASSERT_LOG("Setting a new objective in GPU context");
  m_objective = objective;
  refresh_objective();
  RFASSERT_LOG("Last ran evaluation set to `Not evaluation run`");
  m_lastRanEvaluation = not_eval_run;
}

void RafkoGPUContext::set_weight_updater(rafko_gym::Weight_updaters updater){
  RFASSERT_LOG("Setting weight updater in GPU context to {}", rafko_gym::Weight_updaters_Name(updater));
  m_weightUpdater.reset();
  m_weightUpdater = rafko_gym::UpdaterFactory::build_weight_updater(m_network, updater, *m_settings);
}

void RafkoGPUContext::set_environment(std::shared_ptr<rafko_gym::RafkoEnvironment> environment){
  RFASSERT_LOG("Setting environment in GPU context..");
  RFASSERT_LOG("Environment feature size: {} vs. Network output Neuron number: {}", environment->get_feature_size(), m_network.output_neuron_number());
  RFASSERT(environment->get_feature_size() == m_network.output_neuron_number());
  RFASSERT_LOG("Environment input size: {} vs. Network input size: {}", environment->get_input_size(), m_network.input_data_size());
  RFASSERT(environment->get_input_size() == m_network.input_data_size());
  RFASSERT(static_cast<bool>(m_objective));

  m_environment.reset();
  m_environment = environment;
  std::uint32_t old_output_buffer_num = m_neuronOutputsToEvaluate.size();
  std::uint32_t new_output_buffer_num = m_settings->get_max_processing_threads() * m_environment->get_sequence_size() + 1u;
  m_neuronOutputsToEvaluate.resize(new_output_buffer_num);
  if(old_output_buffer_num < new_output_buffer_num){
    for(std::uint32_t buffer_index = old_output_buffer_num-1; buffer_index < new_output_buffer_num; ++buffer_index){
      m_neuronOutputsToEvaluate[buffer_index].resize(m_environment->get_feature_size());
    }
  }
  m_neuronOutputsToEvaluate.back().resize(m_environment->get_number_of_label_samples());
  RFASSERT_LOG(
    "Agent sequence parameters: {} sequences; {} sequence_size; {} prefill inputs",
    m_environment->get_number_of_sequences(),
    m_environment->get_sequence_size(),
    m_environment->get_prefill_inputs_number()
  );
  m_agent->set_sequence_params(
    m_environment->get_number_of_sequences(),
    m_environment->get_sequence_size(),
    m_environment->get_prefill_inputs_number()
  );
  m_solutionPhase.set_strategy(m_agent);
  upload_weight_table_to_device();
  refresh_objective();
  RFASSERT_LOG("Last ran evaluation set to `Not evaluation run`");
  m_lastRanEvaluation = not_eval_run;
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

  RFASSERT( (start_index_inside_sequence + sequence_truncation) <= m_environment->get_sequence_size() );
  RFASSERT( 0u < sequence_truncation );

  const std::uint32_t truncation_starts_in_sequence = (m_environment->get_prefill_inputs_number() + start_index_inside_sequence);
  std::vector<cl::Event> events(elements_to_upload);
  for(std::uint32_t sequence_index = 0; sequence_index < sequences_to_upload; ++sequence_index){
    for(std::uint32_t label_index = 0; label_index < (m_environment->get_sequence_size() + m_environment->get_prefill_inputs_number()); ++label_index){
      /* add neuron_array_offset */
      byte_offset_solution_phase += ( (m_network.neuron_array_size() - m_network.output_neuron_number()) * sizeof(double) );
      if((truncation_starts_in_sequence <= label_index)&&(label_index < (truncation_starts_in_sequence + sequence_truncation))){
        RFASSERT_LOG(
          "copying from agent[{}] to objective[{} + {}] / {} bytes..",
          byte_offset_solution_phase, byte_offset_error_phase,
          (m_network.output_neuron_number() * sizeof(double)), m_objective->get_input_shapes()[0].get_byte_size<double>()
        );
        return_value = m_openclQueue.enqueueCopyBuffer( /* Upload sequence */
          m_solutionPhase.get_output_buffer() /*src*/, m_errorPhase.get_input_buffer() /*dst*/,
          byte_offset_solution_phase/*src_offset*/, byte_offset_error_phase/*dst_offset*/,
          (m_network.output_neuron_number() * sizeof(double))/*size*/,
          NULL /*events to wait for*/, &events[(sequence_index * sequence_truncation) + label_index - truncation_starts_in_sequence]
        );
        RFASSERT( return_value == CL_SUCCESS );

        byte_offset_error_phase += (m_network.output_neuron_number() * sizeof(double));
      }
      byte_offset_solution_phase += (m_network.output_neuron_number() * sizeof(double));
    }
  }
  return events;
}

double RafkoGPUContext::error_post_process(double raw_error, std::uint32_t labels_evaluated){
  double error_value = raw_error;
  double divisor = std::max(labels_evaluated, 1u);
  double performance_error = m_solutionPhase.acquire_output(
    1u, m_agent->get_output_shapes()[0][0] /* first output, after the size of the first output */
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
  RFASSERT_LOG("Full evaluation in GPU Context..");
  RFASSERT(static_cast<bool>(m_objective));


  if(m_lastRanEvaluation != full_eval_run){
    /* upload mode info */
    cl::Event fill_event;
    return_value = m_openclQueue.enqueueFillBuffer<double>(
      m_solutionPhase.get_input_buffer(), (0.0)/*the double value*/,
      0u /*offset*/, sizeof(double)/*size(bytes)*/,
      NULL/*events to wit for*/, &fill_event
    );
    RFASSERT( return_value == CL_SUCCESS );
    return_value = fill_event.wait();
    RFASSERT( return_value == CL_SUCCESS );

    std::vector<cl::Event> input_events = m_environment->upload_inputs_to_buffer(
      m_openclQueue, m_solutionPhase.get_input_buffer(),
      sizeof(double) * (m_deviceWeightTableSize + m_agent->get_input_shapes()[0][0])/*buffer_start_byte_offset*/,
      0u/*sequence_start_index*/, 0u/*buffer_sequence_start_index*/,
      m_environment->get_number_of_sequences()/*sequences_to_upload*/
    );

    label_events = m_environment->upload_labels_to_buffer(
      m_openclQueue, m_errorPhase.get_input_buffer(), (
        m_environment->get_number_of_label_samples()
        * m_environment->get_feature_size() * sizeof(double)
      ) /*buffer_start_byte_offset*/,
      0u/*sequence_start_index*/, 0/*buffer_sequence_start_index*/,
      m_environment->get_number_of_sequences()/*sequences_to_upload*/,
      0u/*start_index_inside_sequence*/, m_environment->get_sequence_size()/*sequence_truncation*/
    );

    for(cl::Event& input_event : input_events){
      return_value = input_event.wait();
      RFASSERT( return_value == CL_SUCCESS );
    }
  }

  /* run feature phase */
  cl::EnqueueArgs enque_arguments = std::make_from_tuple<cl::EnqueueArgs>(
    std::tuple_cat(std::tie(m_openclQueue), m_agent->get_solution_space())
  );
  m_solutionPhase( enque_arguments );

  /* upload agent output into error phase inputs */
  std::vector<cl::Event> features_events = upload_agent_output(
    m_environment->get_number_of_sequences(),
    0u/*start_index_inside_sequence*/, m_environment->get_sequence_size()/*sequence_truncation*/
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
    std::tuple_cat(std::tie(m_openclQueue), m_objective->get_solution_space())
  );
  m_errorPhase( error_enque_arguments );

  RFASSERT_LOG("Last ran evaluation set to `Full evaluation run`");
  m_lastRanEvaluation = full_eval_run;
  return -error_post_process(m_errorPhase.acquire_output(1u)[0], m_environment->get_number_of_label_samples());
}

double RafkoGPUContext::stochastic_evaluation(bool to_seed, std::uint32_t seed_value){
  [[maybe_unused]]cl_int return_value;
  std::vector<cl::Event> input_events;
  std::vector<cl::Event> label_events;
  RFASSERT_SCOPE(GPU_STOCHASTIC_EVALUATION);
  RFASSERT_LOG("Stochastic evaluation in GPU Context..");

  if(to_seed){
    srand(seed_value);
    RFASSERT_LOG("Seeded run: last used seed: {}; current seed: {}", m_lastUsedSeed, seed_value);
  }
  const std::uint32_t used_minibatch_size = std::min(m_settings->get_minibatch_size(), m_environment->get_number_of_sequences());
  const std::uint32_t used_sequence_truncation = std::min( m_settings->get_memory_truncation(), m_environment->get_sequence_size() );
  const std::uint32_t start_index_inside_sequence = ( rand()%(m_environment->get_sequence_size() - used_sequence_truncation + 1) );
  RFASSERT_LOG(
    "Used minibatch size: {}; sequence_truncation: {}; start index inside sequence: {}",
    used_minibatch_size, used_sequence_truncation, start_index_inside_sequence
  );
  if(
    (m_lastRanEvaluation != random_eval_run)
    ||(m_lastUsedSeed != seed_value)
    ||(!m_lastRandomEvalWasSeeded)
  ){
    cl::Event fill_event;
    RFASSERT_LOG("Updating evaluation buffer..");
    return_value = m_openclQueue.enqueueFillBuffer<double>( /* upload mode info */
      m_solutionPhase.get_input_buffer(), (0)/*the double value*/,
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
      std::uint32_t sequence_start_index = rand()%(m_environment->get_number_of_sequences() - sequences_to_upload + 1u);
      RFASSERT_LOG("Uploading {} sequences starting from {}", sequences_to_upload, sequence_start_index);
      std::vector<cl::Event> input_events = m_environment->upload_inputs_to_buffer(
        m_openclQueue, m_solutionPhase.get_input_buffer(),
        sizeof(double) * (m_deviceWeightTableSize + m_agent->get_input_shapes()[0][0])/*buffer_start_byte_offset*/,
        sequence_start_index, uploaded_sequences/*buffer_sequence_start_index*/,
        sequences_to_upload/*sequences_to_upload*/
      );
      input_events.insert(
        input_events.end(),
        input_events.begin(), input_events.end()
      );
      std::vector<cl::Event> label_events = m_environment->upload_labels_to_buffer(
        m_openclQueue, m_errorPhase.get_input_buffer(), (
          m_environment->get_number_of_label_samples() * m_environment->get_feature_size() * sizeof(double)
        ) /*buffer_start_byte_offset*/,
        sequence_start_index, uploaded_sequences/*buffer_sequence_start_index*/,
        sequences_to_upload/*sequences_to_upload*/,
        start_index_inside_sequence, used_sequence_truncation
      );
      label_events.insert(
        label_events.end(),
        label_events.begin(), label_events.end()
      );
      uploaded_sequences += sequences_to_upload;
    }/*while(uploaded_sequences < used_minibatch_size)*/
  }

  if(to_seed){
    m_lastUsedSeed = seed_value;
    m_lastRandomEvalWasSeeded = true;
  }

  for(cl::Event& event : input_events){
    return_value = event.wait();
    RFASSERT( return_value == CL_SUCCESS );
  }

  /* run feature phase */
  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> sol_space = m_agent->get_solution_space();
  std::get<1>(sol_space) = cl::NDRange(used_minibatch_size);
  cl::EnqueueArgs enque_arguments = std::make_from_tuple<cl::EnqueueArgs>(
    std::tuple_cat(std::tie(m_openclQueue), sol_space)
  );
  m_solutionPhase( enque_arguments );

  /* upload agent output into error phase inputs */
  std::vector<cl::Event> features_events = upload_agent_output(
    used_minibatch_size, start_index_inside_sequence, used_sequence_truncation
  );

  /* fill the rest of the output buffer with the label value */
  cl::Event fill_event;
  std::uint32_t uploaded_bytes_count = (m_environment->get_feature_size() * used_sequence_truncation * used_minibatch_size * sizeof(double));
  std::uint32_t minibatch_labels_byte_size = (m_environment->get_feature_size() * m_environment->get_sequence_size() * used_minibatch_size * sizeof(double));
  RFASSERT_LOG(
    "Copying label values[{}] to agent output buffer[{} + {}] as dummy data..",
    minibatch_labels_byte_size, uploaded_bytes_count, (minibatch_labels_byte_size - uploaded_bytes_count)
  );

  return_value = m_openclQueue.enqueueCopyBuffer(
    m_errorPhase.get_input_buffer()/*src*/,  m_errorPhase.get_input_buffer()/*dst*/,
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
  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> error_sol_space = m_objective->get_solution_space();
  std::get<1>(error_sol_space) = cl::NDRange(used_minibatch_size * used_sequence_truncation);
  cl::EnqueueArgs error_enque_arguments = std::make_from_tuple<cl::EnqueueArgs>(
    std::tuple_cat(std::tie(m_openclQueue), error_sol_space)
  );
  m_errorPhase( error_enque_arguments );
  RFASSERT_LOG("Last ran evaluation set to `Random evaluation run`");
  m_lastRanEvaluation = random_eval_run;
  return -error_post_process(
    m_errorPhase.acquire_output(1u)[0], (used_minibatch_size * m_environment->get_sequence_size())
  );
}

rafko_utilities::ConstVectorSubrange<> RafkoGPUContext::solve(
  const std::vector<double>& input, bool reset_neuron_data, std::uint32_t thread_index
){
  RFASSERT_SCOPE(GPU_STANDALONE_SOLVE);
  RFASSERT_LOG("Solving network in GPU Context..");
  RFASSERT_LOG("Thread index in solve: {}", thread_index);
  RFASSERT(0 == thread_index);
  if(0u != thread_index)
    throw std::runtime_error("Multi-threaded openCL Environment not supported!");

  [[maybe_unused]]cl_int return_value;
  cl::Event fill_event;
  const std::uint32_t network_memory_slots = std::max(2u, (m_network.memory_size() - 1u));
  const std::size_t network_used_bytes = sizeof(double) * network_memory_slots * m_network.neuron_array_size();

  if(reset_neuron_data || (m_lastRanEvaluation != not_eval_run) ){
    RFASSERT_LOG("Not resetting agent data..");
    return_value = m_openclQueue.enqueueFillBuffer<double>(
      m_solutionPhase.get_output_buffer(), (0.0)/* the data(pattern) value */,
      0u/*offset*/, network_used_bytes/*size*/,
      NULL/*events to wait for*/, &fill_event
    );
    RFASSERT( return_value == CL_SUCCESS );

    return_value = fill_event.wait();
    RFASSERT( return_value == CL_SUCCESS );
  }else{ /* Neuron memory not resetted, keep network memory consistent */
    const std::uint32_t network_memory_span_bytes = m_network.neuron_array_size() * sizeof(double);
    RFASSERT_LOG(
      "Resetting agent data; Memory slots: {}; Memory size in bytes: {}",
      network_memory_slots, network_memory_span_bytes
    );
    for(std::uint32_t memory_slot = 0; memory_slot < (network_memory_slots-1u); ++memory_slot){
      return_value = m_openclQueue.enqueueCopyBuffer(
        m_solutionPhase.get_output_buffer()/*src*/, m_solutionPhase.get_output_buffer() /*dst*/,
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
  return_value = m_openclQueue.enqueueFillBuffer<double>(
    m_solutionPhase.get_input_buffer(), (69.420)/*the double value*/,
    0u /*offset*/, sizeof(double)/*size(bytes)*/,
    NULL/*events to wit for*/, &fill_event
  );
  RFASSERT( return_value == CL_SUCCESS );
  return_value = fill_event.wait();
  RFASSERT( return_value == CL_SUCCESS );

  /* upload inputs */
  return_value = m_openclQueue.enqueueWriteBuffer(
    m_solutionPhase.get_input_buffer(), CL_TRUE,
    (sizeof(double) * (m_deviceWeightTableSize + 1u))/*offset: mode and weights*/,
    (sizeof(double) * input.size())/*size*/,
    input.data(), NULL
  );
  RFASSERT( return_value == CL_SUCCESS );

  std::tuple<cl::NDRange,cl::NDRange,cl::NDRange> sol_space = m_agent->get_solution_space();
  std::get<1>(sol_space) = cl::NDRange(1);
  cl::EnqueueArgs enq = std::make_from_tuple<cl::EnqueueArgs>( std::tuple_cat(std::tie(m_openclQueue), sol_space) );
  m_solutionPhase( enq );

  std::uint32_t output_array_start = ( /* the end of the last memory slot contains the network data */
    (std::max(2u, m_network.memory_size()) * m_network.neuron_array_size()) - m_network.output_neuron_number()
  );
  RFASSERT_LOG("Output array start: {}", output_array_start);
  if(static_cast<std::int32_t>(m_standaloneSolutionResult.size()) == m_network.neuron_array_size()){
    RFASSERT_LOG("Loading output to already allocated vector..");
    m_solutionPhase.load_output(m_standaloneSolutionResult.data(), m_network.output_neuron_number(), output_array_start);
  }else{
    RFASSERT_LOG("Acquiring output and moving it to new vector..");
    std::unique_ptr<double[]> output_ptr = m_solutionPhase.acquire_output(m_network.output_neuron_number(), output_array_start);
    m_standaloneSolutionResult = std::vector<double>(output_ptr.get(), output_ptr.get() + m_network.output_neuron_number());
  }
  RFASSERT_LOGV(m_standaloneSolutionResult, "Resulting output:");

  RFASSERT_LOG("Last ran evaluation set to `Not evaluation run`");
  m_lastRanEvaluation = not_eval_run;

  return { m_standaloneSolutionResult.end() - m_network.output_neuron_number(), m_standaloneSolutionResult.end() };
}

} /* namespace rafko_mainframe */
