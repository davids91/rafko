package services;

import io.grpc.Channel;
import io.grpc.StatusRuntimeException;
import org.rafko.mainframe.RafkoDeepLearningService;
import org.rafko.mainframe.Rafko_deep_learningGrpc;
import org.rafko.sparse_net_library.RafkoCommon;
import org.rafko.sparse_net_library.RafkoSparseNet;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RafkoDLClient {
    private static final Logger logger = Logger.getLogger(services.RafkoDLClient.class.getName());
    private Rafko_deep_learningGrpc.Rafko_deep_learningBlockingStub server_stub;
    private Runnable on_disconnect;

    public RafkoDLClient(Channel channel, Runnable on_disconnect_){
        server_stub = Rafko_deep_learningGrpc.newBlockingStub(channel);
        on_disconnect = on_disconnect_;
    }

    public RafkoDeepLearningService.Slot_response ping(){
        /* if any commend is successful, then server is online! */
        RafkoDeepLearningService.Slot_request request = RafkoDeepLearningService.Slot_request
            .newBuilder().setTargetSlotId("MOOT").build();
        try {
            return server_stub.ping(request);
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
        } catch(Exception e){
            e.printStackTrace();
            on_disconnect.run();
        }
        return null;
    }

    public RafkoDeepLearningService.Slot_response add_server_slot(
        RafkoDeepLearningService.Service_slot attempt
    ) throws StatusRuntimeException{
        try {
            return server_stub.addSlot(attempt);
        } catch (StatusRuntimeException e){
            e.printStackTrace();
            on_disconnect.run();
            throw e;
        }
    }

    public RafkoDeepLearningService.Slot_response update_server_slot(
        RafkoDeepLearningService.Service_slot service_slot
    ) throws StatusRuntimeException{
        try {
            return server_stub.updateSlot(service_slot);
        } catch (StatusRuntimeException e){
            e.printStackTrace();
            on_disconnect.run();
            throw e;
        }
    }

    public RafkoSparseNet.SparseNet get_network(
            String slot_id
    ) throws StatusRuntimeException{
        try {
            RafkoDeepLearningService.Slot_request get_network_request = RafkoDeepLearningService.Slot_request.newBuilder()
                    .setTargetSlotId(slot_id)
                    .build();
            return server_stub.getNetwork(get_network_request);
        } catch (StatusRuntimeException e){
            e.printStackTrace();
            on_disconnect.run();
            throw e;
        }
    }

    public RafkoSparseNet.SparseNet create_network(
        String slot_id, int input_size, double expected_input_range,
        ArrayList<RafkoCommon.transfer_functions> allowed_transfer_functions, ArrayList<Integer> layer_sizes
    ) throws StatusRuntimeException{
        try {
            System.out.print(",,,");
            RafkoDeepLearningService.Build_network_request build_request = RafkoDeepLearningService.Build_network_request.newBuilder()
                    .setTargetSlotId(slot_id).setInputSize(input_size).setExpectedInputRange(expected_input_range)
                    .addAllAllowedTransfersByLayer(allowed_transfer_functions)
                    .addAllLayerSizes(layer_sizes)
                    .build();

            System.out.print("...");
            RafkoDeepLearningService.Slot_response answer = server_stub.buildNetwork(build_request);

            System.out.println("Built Network in: " + answer.getSlotId());

            return get_network(slot_id);
        } catch (StatusRuntimeException e){
            e.printStackTrace();
            on_disconnect.run();
            throw e;
        }
    }

}
