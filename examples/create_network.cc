#include <iostream>
#include <memory>
#include <cmath>
#include <thread>
#include <iostream>
#include <fstream>

#include <rafko_protocol/rafko_net.pb.h>
#include <rafko.hpp>

int main(){
  /* Create a Settings Object. This stores to essential parameters for almost everything inside the Rafko Framework */
  std::shared_ptr<rafko_mainframe::RafkoSettings> settings = std::make_shared<rafko_mainframe::RafkoSettings>();

  /* Create a Densely Connected Neural Network */
  using rafko_net::RafkoNet;
  std::unique_ptr<RafkoNet> network = std::unique_ptr<RafkoNet>(
    rafko_net::RafkoNetBuilder(*settings)
    /* +++ Basic Parameters to set +++ */
    .input_size(4) /* The number of values the network reads as its input vector */

    /* +++ Optional Structural Parameters to set +++ */
    .set_neuron_input_function(0u/*layer_index*/, 0u/*layer_neuron_index*/, rafko_net::input_function_multiply) /* Explicitly set the input function of a Neuron inside the layer */
    .set_neuron_transfer_function(0u, 1u, rafko_net::transfer_function_relu) /* Explicitly set the transfer function of a Neuron inside the layer */
    /*!Note: If not set explicitly, the builder will select a random transfer function with the arguments set from @allowed_transfer_functions_by_layer */
    .set_neuron_spike_function(0u, 2u, rafko_net::spike_function_none) /* Explicitly set the spike function of a Neuron inside the layer */
    .allowed_transfer_functions_by_layer({ /* Set the allowed transfer functions for each layer */
      {rafko_net::transfer_function_relu, rafko_net::transfer_function_selu},
      {rafko_net::transfer_function_selu},
      {rafko_net::transfer_function_selu}
    })
    /*!Note: If not set explicitly, a random random transfer function will be selected without restrictions */
    .layer_input_convolution(0u) /*!Note: to not have a fully connected layers, but convolutions the following can be used */
      .kernel_size(1,1).kernel_stride(1,1).input_padding(0,0)
      /*Note: Input or output dimensions both need not be added, but compared to layer sizes when building */
      .input_size(2,2).output_size(2,2)
      .validate() /* validation is neccesary where the sizes are compared, so it is made sure that the input and output sizes do not go out of bounds */
      /* .. which returns with a reference of the actual Builder, s othe chaining can be resumed */
      /*!Note: sorry I have trust issues */

    /* +++ Functional Parameters to set +++ */
    .expected_input_range(1.0) /* The expected maximum value for **each** input value */
    .add_feature_to_layer(0u/*layer_index*/, rafko_net::neuron_group_feature_softmax) /* If set, after each run, the sum for each neurons activation value in that layer will equal 1.0 */
    .add_feature_to_layer(0u, rafko_net::neuron_group_feature_dropout_regularization) /* To improve training roboustness, some neuron activation values of this layer are randomly set to 0.0 */
    /*!Note: uses the setting @RafkoSettings::get_dropout_probability */
    .add_feature_to_layer(0u, rafko_net::neuron_group_feature_l2_regularization) /* Adds Lasso Regression Terms for the training for the selected layer */
    .add_feature_to_layer(1u, rafko_net::neuron_group_feature_l1_regularization) /* Adds Ridge Regression Terms for the training for the selected layer */

    /* +++ Neural Memory Parameters to set +++ */
    .add_feature_to_layer(1u, rafko_net::neuron_group_feature_boltzmann_knot) /* If set, each Neuron will take its layers layers activation values as inputs. */
    /*!Note: the inputs recursing from the layer are taken from a previous run, with the values equaling 0.0 in the first run */
    .add_neuron_recurrence(2u/*layer_index*/, 0u/*layer_neuron_index*/, 1u/*past_value*/) /* Add Neuron it's own activation value from the previous @past_index-th run of the network  */

    /* +++ Actual Build +++ */
    .create_layers({4, 3, 1}) /* The provided vector describes the number of neurons in each layer */
    /*!Note: As an optional argument the parameter @allowed_transfer_functions_by_layer can also be added here */
  );

  /*!Note: Rafko May also use Protocol Buffers Arena implementation, which means that the object returned by the builder is owned by it.
   * Because of this, it is perfectly acceptable to build a Network like in the below example, because the declared arena will have ownership of it.
   */
  google::protobuf::Arena everything_storage;
  settings->set_arena_ptr(&everything_storage);
  RafkoNet& first_network_on_arena = *rafko_net::RafkoNetBuilder(*settings).input_size(2).create_layers({3, 3, 1});

  /*!Note: Because of this, every new Network is placed into the arena; The below will be the second instance of a network.
   * Replacing an already allocated Network inside the arena with a newly built one is demonstrated below.
   */
  RafkoNet* additional_network_on_arena = rafko_net::RafkoNetBuilder(*settings).input_size(2).create_layers({3, 5, 1});

  /* Swap the network with a bigger one*/
  rafko_net::RafkoNetBuilder(*settings)
  .input_size(2) /* The number of values the network reads as its input vector */
  .build_create_layers_and_swap(additional_network_on_arena, {5, 5, 1});

  /*!Note: Even with an existing Arena updated in settings, it is possible to create objects on the Heap.
   * The below function call returns ownership to the caller
   */
  std::unique_ptr<RafkoNet> another_managed_network = std::unique_ptr<RafkoNet>(
    rafko_net::RafkoNetBuilder(*settings).input_size(2).create_layers(nullptr, {2, 3, 1})
  );

  /*!Note: It is possible to manually modify a network, build connections and Add/Remove Neurons.
   * below a two Neuron network is compiled, the first accepts the input, and the last (the output Neuron)
   * accepts the output of the first Neuron as well as the value of itself in the previous 3rd run of the network.
   */
  first_network_on_arena.set_memory_size(4u); /* Network memory size needs to be +1 to account for the actual loop in progress */
  first_network_on_arena.set_input_data_size(2u);
  first_network_on_arena.set_output_neuron_number(1u);
  first_network_on_arena.mutable_neuron_group_features()->Clear();
  first_network_on_arena.mutable_neuron_array()->Clear();
  first_network_on_arena.mutable_weight_table()->Clear();

  /* +++ set the data of the first Neuron +++ */
  rafko_net::Neuron& first_neuron = *first_network_on_arena.add_neuron_array();
  first_neuron.set_input_function(rafko_net::input_function_multiply);
  first_neuron.set_transfer_function(rafko_net::transfer_function_elu);
  first_neuron.set_spike_function(rafko_net::spike_function_none);

  /*!Note: rafko_net::Neuron::input_weights sets the number of weights attached to the Neuron, while
   * rafko_net::Neuron::input_indices determines the input values a Neuron takes. the latter must always be
   * less or equal, than the former. If there are more input weights, than inputs for a given Neuron, the
   * remaining weights behave as bias values. There can be multiple bias values assigned to one Neuron,
   * their values are collected by the Neurons input function.
   */
  rafko_net::InputSynapseInterval& first_neuron_inputs = *first_neuron.add_input_indices();
  first_neuron_inputs.set_starts(-1); /* negative numbers mean the input synapse takes its value externally. In this case, from outside the network. */
  first_neuron_inputs.set_interval_size(2); /* equals the number of inputs the network has, based on the network structure */
  rafko_net::IndexSynapseInterval& first_neuron_weights = *first_neuron.add_input_weights();
  first_neuron_weights.set_starts(0u); /* there are no negative start numbers in rafko_net::IndexSynapseInterval */
  first_neuron_weights.set_interval_size(1 + 2 + 1); /* 1 weight for the spike function, 2 for the inputs and 1 for the bias value */
  for(std::uint32_t weight_count = 0; weight_count < 4; ++weight_count){
    first_network_on_arena.add_weight_table(rand()%10 * 0.1);
  }

  /* +++ set the data of the second Neuron +++ */
  rafko_net::Neuron& second_neuron = *first_network_on_arena.add_neuron_array();
  second_neuron.set_input_function(rafko_net::input_function_multiply);
  second_neuron.set_transfer_function(rafko_net::transfer_function_elu);
  second_neuron.set_spike_function(rafko_net::spike_function_none);

  rafko_net::InputSynapseInterval& second_neuron_present_input = *second_neuron.add_input_indices();
  second_neuron_present_input.set_starts(0); /* positive numbers mean the input synapse takes its value internally. In this case, from inside the network's Neuron values. */
  second_neuron_present_input.set_interval_size(1); /* this input synapse spans only one Neuron */

  /*!Note: Each Neuron can have multiple synapses. As a kind of compression, the ranges are stored, so it is optimal to have
   * as few "fragmentations" as possible with connections, to spare the number of synapse elements to store inside the network.
   */
  rafko_net::InputSynapseInterval& second_neuron_past_input = *second_neuron.add_input_indices();
  second_neuron_past_input.set_starts(1);   /* Neuron[1] --> meaning itself */
  second_neuron_past_input.set_interval_size(1);
  second_neuron_past_input.set_reach_past_loops(3);/* meaning: from the 3rd past run of the Neuron */

  rafko_net::IndexSynapseInterval& second_neuron_weights_0 = *second_neuron.add_input_weights();
  second_neuron_weights_0.set_starts(4u); /* The first weight after the previously added weights */
  second_neuron_weights_0.set_interval_size(1u + 2u); /* this weight synapse spans through the spike function and inputs */

  rafko_net::IndexSynapseInterval& second_neuron_weights_1 = *second_neuron.add_input_weights();
  second_neuron_weights_1.set_starts(3u); /* this synapse points to the previous Neurons bias value. Meaning weights can be shared this way */
  second_neuron_weights_1.set_interval_size(1u);

  for(std::uint32_t weight_count = 0; weight_count < 3; ++weight_count){ /*!Note: only 3 weights need to be added */
    first_network_on_arena.add_weight_table(rand()%10 * 0.1);
  }
  /*!Note: There's a reason rafko_net::RafkoNetBuilder was implemented for the creation of networks.. :)
   * However, tinkering is good and encouraged! Make sure to catch any exceptions the library might throw!
   */

  /*!Note: To save a network to a file, or any stream std C++ functionality is available */
  std::filebuf file_buffer;
  file_buffer.open("network.rfnet", std::ios::out);
  std::ostream os(&file_buffer);
  network->SerializeToOstream(&os);
  file_buffer.close();

  file_buffer.open("network.rfnet", std::ios::in);
  std::istream is(&file_buffer);
  network->ParseFromIstream(&is);
  file_buffer.close();

  google::protobuf::ShutdownProtobufLibrary(); /* This is only needed to avoid false positive memory leak reports in memory leak analyzers */
  return 0;
}
