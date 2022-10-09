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

#ifndef RAFKO_ENVIRONMENT_H
#define RAFKO_ENVIRONMENT_H

#include "rafko_global.h"

#include <memory>

#include "rafko_protocol/rafko_net.pb.h"
#include "rafko_protocol/training.pb.h"
#include "rafko_utilities/models/data_pool.h"

namespace RAFKO_FULL_EXPORT rafko_gym{

/**
 * @brief      A class representing one set of changes inside a network during training
 */
class RAFKO_FULL_EXPORT RafkoNetworkDeltaChainLink{
public:
  RafkoNetworkDeltaChainLink(
    const rafko_net::RafkoNet& original_network_, NetworkDeltaChainLinkData data_ = {},
    std::shared_ptr<RafkoNetworkDeltaChainLink> parent_ = std::shared_ptr<RafkoNetworkDeltaChainLink>()
  ):original_network(original_network_)
  , parent(parent_)
  , data(data_)
  { }

  const rafko_net::RafkoNet& get_original_network(){
    return original_network;
  }

  rafko_net::RafkoNet get_current_network();

  std::pair<std::unique_ptr<rafko_net::RafkoNet>,RafkoNetworkDeltaChainLink> create_new_chain(){
    std::unique_ptr<rafko_net::RafkoNet> current_network = std::make_unique<rafko_net::RafkoNet>(get_current_network());
    rafko_net::RafkoNet* current_network_ptr = current_network.get();
    return std::make_pair( std::move(current_network), RafkoNetworkDeltaChainLink(*current_network_ptr) );
  }

  void store_change(std::uint32_t weight_index, double weight_delta){
    if(!is_last_change_simple())
      data.add_simple_changes()->set_version(get_latest_version());
    apply_change(weight_index, weight_delta, *data.mutable_simple_changes(data.simple_changes_size() - 1)->mutable_weights_delta());
    network_built = false;
  }
  void store_change(std::vector<double>& weight_delta);
  void store_change(NetworkWeightVectorDelta&& weight_delta);
  void store_change(NonStructuralNetworkDelta&& change);
  void store_change(StructuralNetworkDelta&& change);

  std::uint32_t get_latest_version() const{
    std::uint32_t version = 0u;
    if( (0 == (data.simple_changes_size() + data.structural_changes_size())) && parent )
      version = parent->get_latest_version();
    if(0 < data.simple_changes_size())
      version = data.simple_changes(data.simple_changes_size() - 1u).version();
    if(0 < data.structural_changes_size())
      version = std::max( version, data.structural_changes(data.structural_changes_size() - 1u).version() );
    return version;
  }

  static void apply_to_network(NetworkDeltaChainLinkData& delta, rafko_net::RafkoNet& network);
  static void apply_change(const NonStructuralNetworkDelta& change, rafko_net::RafkoNet& network);
  static void apply_change(std::uint32_t weight_index, double weight_delta, NetworkWeightVectorDelta& weights_delta);

private:
  static rafko_utilities::DataPool<> tmp_data_pool;
  const rafko_net::RafkoNet& original_network;
  std::shared_ptr<RafkoNetworkDeltaChainLink> parent;
  NetworkDeltaChainLinkData data;

  bool network_built = false;
  bool network_structure_built = false;
  rafko_net::RafkoNet current_network = rafko_net::RafkoNet();

  bool is_last_change_simple() const{
    return (
      (0 < data.simple_changes_size())
      &&((0u == data.structural_changes_size())
      ||(
        data.structural_changes(data.structural_changes_size() - 1u).version()
        <= data.simple_changes(data.simple_changes_size() - 1u).version()
      )
    ));
  }

  /**
   * @brief      Insert an element to the given position into the given field by
   *             first adding it to the end, and then reverse iterating and swapping elements
   *             until the desired position is reached
   *
   * @param      message_field  The message field
   * @param[in]  value          The value
   * @param[in]  position       The position
   */
  static void insert_element_at_position(google::protobuf::RepeatedField<double>& message_field, double value, std::uint32_t position){
    *message_field.Add() = value;
    for(std::int32_t i(message_field.size() - 1); i > static_cast<std::int32_t>(position); --i)
      message_field.SwapElements(i, i - 1);
  }

  static void unwrap_change_to(std::vector<double>& vector, const NetworkWeightVectorDelta& delta);

};

} /* namespace rafko_gym */

#endif /* RAFKO_ENVIRONMENT_H */
