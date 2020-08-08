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

#ifndef SERVER_SLOT_H
#define SERVER_SLOT_H

#include "sparse_net_global.h"

#include <string>
#include <memory>

#include "gen/sparse_net.pb.h"
#include "gen/deep_learning_service.pb.h"
#include "sparse_net_library/models/data_aggregate.h"

namespace rafko_mainframe{

using std::string;
using std::shared_ptr;

using sparse_net_library::Data_aggregate;
using sparse_net_library::SparseNet;

/**
 * @brief      This class describes a common ancestor and interface of whatever 
 *             is able to run in a @DeepLearning_server.
 */
class Server_slot{
public: 
  Server_slot(void)
  : service_slot()
  { service_slot.set_slot_id(generate_uuid()); }
  virtual ~Server_slot() = default;

  /**
   * @brief      Initializes based on the provided backbone
   *
   * @param[in]  service_slot_  The service slot
   */
  virtual void initialize(Service_slot&& service_slot_) = 0;

  /**
   * @brief      The main loop of the server to run to be able to provide the service
   */
  virtual void loop(void) = 0;

  /**
   * @brief      Resets the object.
   */
  virtual void reset(void) = 0;

  /**
   * @brief      Update the currently loaded network with the provided one
   *
   * @param[in]  net_   The network to overwrite the current one
   */
  virtual void update_network(SparseNet&& net_) = 0;

  /**
   * @brief      Accept the request provided in the argument. Implementation may vary.
   *
   * @param[in]  request_bitstring  The requests packed into a bitstring
   */
  virtual void accept_request(uint32 request_bitstring) = 0;

  /**
   * @brief      Runs the attached network if it's valied
   *
   * @param      data_stream  The input data stream
   *
   * @return     The output data stream.
   */
  virtual Neural_io_stream run_net_once(const Neural_io_stream& data_stream) = 0;

  /**
   * @brief      Gets a sample from the attached training dataset
   *
   * @param[in]  index  The index of the sample to get
   *
   * @return     The training sample as a packed stream: First the input, then the label.
   */
  virtual Neural_io_stream get_training_sample(uint32 sample_index, bool get_input, bool get_label) const = 0;

  /**
   * @brief      Gets a sample from the attached testing dataset
   *
   * @param[in]  index  The index of the sample to get
   *
   * @return     The testing sample as a packed stream: First the input, then the label.
   */
  virtual Neural_io_stream get_testing_sample(uint32 sample_index, bool get_input, bool get_label) const = 0;

  /** 
   * @brief      Queries relevant information about the @Server_slot.
   *
   * @param[in]  request  uses @request_bitstring to ask for @Slot_info_field values
   *
   * @return     Information packets returned in the order given by @Slot_info_field.
   */
  virtual Slot_info get_info(uint32 request_bitstring) = 0;

  /**
   * @brief      Provide the loaded network
   *
   * @return     The network currently loaded in the configuration
   */
  virtual SparseNet get_network(void) const = 0;

  /**
   * @brief      Gets the identifier of the slot
   *
   * @return     The uuid.
   */
  string get_uuid(void) const;

  /**
   * @brief      Provides the status of the server slot.
   *
   * @return     The status, described in the file @proto/deep_learning_service.proto
   */
  Slot_response get_status(void) const;

protected:
  Service_slot service_slot;

  /**
   * @brief      Generates a unique identifier, with a guarantee that the currently 
   *             saved slot Identifiers shall be left out.
   *
   * @return     A random unique Identifier string
   */
  string generate_uuid(void);

  /**
   * @brief      Updates status of the service slot so it shall store the induvidual status bits
   *             instead of the final status value. It's safe to call multiple times.
   */
  void expose_state(void){
    if(SERV_SLOT_OK == service_slot.state())
      service_slot.set_state(0);
  }

  /**
   * @brief      Updates status of the service slot based on the object state.
   *             In order to set the correct state value, the state needs to be exposed
   *             by @expose_state to update the status flags
   */
  void finalize_state(void){
    if(0 == service_slot.state()) /* No issues found, great! */
      service_slot.set_state(SERV_SLOT_OK);
  }

  /**
   * @brief      Utility function to convert a sample of the provided data set into a data stream
   *
   * @param[in]  data_set      The data set to get the sample from
   * @param[in]  sample_index  The sample index
   * @param      target        The target stream to copy the data into. It shall have the
   *                           size components of the requested dataset to sizes matching
   *                           the sizes inside the @data_set to copy the information from.
   *                           Whichever size components are not set correctly shall not be copied
   */
  void get_data_sample(shared_ptr<Data_aggregate> data_set, uint32 sample_index, Neural_io_stream& target) const;

};

} /* namespace rafko_mainframe */

#endif /* SERVER_SLOT_H */
