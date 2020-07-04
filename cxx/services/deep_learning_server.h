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

#ifndef DEEP_LEARNING_SERVER_H
#define DEEP_LEARNING_SERVER_H

#include "sparse_net_global.h"
#include "gen/deep_learning_service.grpc.pb.h"

namespace sparse_net_library{

/**
 * @brief      This class describes a server for deep learning related tasks. The supported operations are described in
 *             the @/proto/deep_learning_services.proto file. Functions defined in the service are thread-safe.
 */
class Deep_learning_server final : public Rafko_deep_learning::Service{
public:
    ~Service();
    ::grpc::Status add_slot(::grpc::ServerContext* context, ::grpc::ServerReader< ::sparse_net_library::Service_slot>* reader, ::sparse_net_library::Slot_status* response);
    ::grpc::Status update_slot(::grpc::ServerContext* context, ::grpc::ServerReader< ::sparse_net_library::Service_slot>* reader, ::sparse_net_library::Slot_status* response);
    ::grpc::Status request_action(::grpc::ServerContext* context, ::grpc::ServerReader< ::sparse_net_library::Slot_request>* reader, ::sparse_net_library::Slot_status* response);
    ::grpc::Status run_net_for(::grpc::ServerContext* context, ::grpc::ServerReaderWriter< ::sparse_net_library::Neural_io_stream, ::sparse_net_library::Slot_request>* stream);

    /**
     * @brief      The main loop of the server to run to be able to provide the service
     */
    void run();

private:
  /* Service slots */

};

} /* namespace sparse_net_library */

#endif /* DEEP_LEARNING_SERVER_H */
