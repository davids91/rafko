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

#include "gen/sparse_net.pb.h"
#include "gen/deep_learning_service.pb.h"

namespace rafko_mainframe{

using std::string;

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
   * @param[in]  request_  The request to be accepted
   */
  virtual void accept_request(Slot_request&& request_) = 0;

  /**
   * @brief      Runs the attached network, if it's valid, and uploads the result in the 
   *             reference
   *
   * @param      data_stream  the data to be taken as input, and the data to upload the result to
   */
  virtual void run_net_once(Neural_io_stream& data_stream) = 0;

  /**
   * @brief      Queries relevant information about the @Server_slot.
   *
   * @param[in]  request  uses @request_bitstring to ask for @Slot_info_field values
   *
   * @return     Information packets returned in the order given by @Slot_info_field.
   */
  virtual Slot_info get_info(Slot_request request) = 0;

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
  virtual string get_uuid(void) const = 0;

  /**
   * @brief      Provides the status of the server slot.
   *
   * @return     The status, described in the file @proto/deep_learning_service.proto
   */
  Slot_response get_status(void);

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

};

} /* namespace rafko_mainframe */

#endif /* SERVER_SLOT_H */
