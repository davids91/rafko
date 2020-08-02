package services;

import io.grpc.Channel;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import org.rafko.mainframe.RafkoDeepLearningService;
import org.rafko.mainframe.Rafko_deep_learningGrpc;
import org.rafko.sparse_net_library.RafkoCommon;
import org.rafko.sparse_net_library.RafkoSparseNet;

import java.util.ArrayList;
import java.util.InvalidPropertiesFormatException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.rafko.mainframe.RafkoDeepLearningService.Slot_action_field.SERV_SLOT_RUN_ONCE_VALUE;
import static org.rafko.mainframe.RafkoDeepLearningService.Slot_action_field.SERV_SLOT_TO_REFRESH_SOLUTION_VALUE;

public class RafkoDLClient {
    private static final Logger logger = Logger.getLogger(services.RafkoDLClient.class.getName());
    private Rafko_deep_learningGrpc.Rafko_deep_learningBlockingStub server_rpc;
    private Rafko_deep_learningGrpc.Rafko_deep_learningStub server_async_rpc;
    private Runnable on_disconnect;

    public RafkoDLClient(Channel channel, Runnable on_disconnect_){
        server_rpc = Rafko_deep_learningGrpc.newBlockingStub(channel);
        server_async_rpc = Rafko_deep_learningGrpc.newStub(channel);
        on_disconnect = on_disconnect_;
    }

    public RafkoDeepLearningService.Slot_response ping(String id){
        /* if any commend is successful, then server is online! */
        RafkoDeepLearningService.Slot_request request = RafkoDeepLearningService.Slot_request
                .newBuilder().setTargetSlotId(id).build();
        try {
            return server_rpc.ping(request);
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
        } catch(Exception e){
            e.printStackTrace();
            on_disconnect.run();
        }
        return null;
    }

    public RafkoDeepLearningService.Neural_io_stream run_net_once(String slot_id, int dataset_index){
        final RafkoDeepLearningService.Slot_response[] response = new RafkoDeepLearningService.Slot_response[1];
        final CountDownLatch finishLatch = new CountDownLatch(1);
        StreamObserver<RafkoDeepLearningService.Slot_request> request = server_async_rpc.requestAction(new StreamObserver<>(){

            @Override
            public void onNext(RafkoDeepLearningService.Slot_response value) {
                response[0] = value;
            }

            @Override
            public void onError(Throwable t) {
                finishLatch.countDown();
            }

            @Override
            public void onCompleted() {
                finishLatch.countDown();
            }
        });
        request.onNext(
                RafkoDeepLearningService.Slot_request.newBuilder()
                        .setTargetSlotId(slot_id).setRequestBitstring( SERV_SLOT_TO_REFRESH_SOLUTION_VALUE | SERV_SLOT_RUN_ONCE_VALUE )
                        .setRequestIndex(dataset_index)
                        .build()
        );
        request.onCompleted();
        try {
            finishLatch.await(1, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            e.printStackTrace();
            on_disconnect.run();
        }
        System.out.println("request finished!");
        return response[0].getDataStream();
    }

    public RafkoDeepLearningService.Neural_io_stream run_net_once(String slot_id, RafkoDeepLearningService.Neural_io_stream input){
        final RafkoDeepLearningService.Slot_response[] response = new RafkoDeepLearningService.Slot_response[1];
        final CountDownLatch finishLatch = new CountDownLatch(1);
        StreamObserver<RafkoDeepLearningService.Slot_request> request = server_async_rpc.requestAction(new StreamObserver<>(){

            @Override
            public void onNext(RafkoDeepLearningService.Slot_response value) {
                response[0] = value;
            }

            @Override
            public void onError(Throwable t) {
                finishLatch.countDown();
            }

            @Override
            public void onCompleted() {
                finishLatch.countDown();
            }
        });
        request.onNext(
                RafkoDeepLearningService.Slot_request.newBuilder()
                .setTargetSlotId(slot_id).setRequestBitstring( SERV_SLOT_TO_REFRESH_SOLUTION_VALUE | SERV_SLOT_RUN_ONCE_VALUE )
                .setDataStream(input)
                .build()
        );
        request.onCompleted();
        try {
            finishLatch.await(1, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            e.printStackTrace();
            on_disconnect.run();
        }
        return response[0].getDataStream();
    }

    public RafkoDeepLearningService.Slot_response request_one_action(String slot_id, int request_bitstring, int request_index){
        final RafkoDeepLearningService.Slot_response[] response = new RafkoDeepLearningService.Slot_response[1];
        final CountDownLatch finishLatch = new CountDownLatch(1);
        StreamObserver<RafkoDeepLearningService.Slot_request> request = server_async_rpc.requestAction(new StreamObserver<>(){

            @Override
            public void onNext(RafkoDeepLearningService.Slot_response value) {
                response[0] = value;
            }

            @Override
            public void onError(Throwable t) {
                finishLatch.countDown();
            }

            @Override
            public void onCompleted() {
                finishLatch.countDown();
            }
        });
        request.onNext(
            RafkoDeepLearningService.Slot_request.newBuilder()
                .setTargetSlotId(slot_id).setRequestBitstring(request_bitstring).setRequestIndex(request_index)
                .build()
        );
        request.onCompleted();
        try {
            finishLatch.await(1, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            e.printStackTrace();
            on_disconnect.run();
        }
        System.out.println("request finished!");
        return response[0];
    }

    public RafkoDeepLearningService.Slot_info get_info(RafkoDeepLearningService.Slot_request request){
        return server_rpc.getInfo(request);
    }

    public RafkoDeepLearningService.Slot_response ping(){
        return ping("MOOT");
    }

    public RafkoDeepLearningService.Slot_response add_server_slot(
        RafkoDeepLearningService.Service_slot attempt
    ) throws StatusRuntimeException{
        try {
            return server_rpc.addSlot(attempt);
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
            return server_rpc.updateSlot(service_slot);
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
            return server_rpc.getNetwork(get_network_request);
        } catch (StatusRuntimeException e){
            e.printStackTrace();
            on_disconnect.run();
            throw e;
        }
    }

    public RafkoSparseNet.SparseNet create_network(
        String slot_id, int input_size, double expected_input_range,
        ArrayList<RafkoCommon.transfer_functions> allowed_transfer_functions, ArrayList<Integer> layer_sizes
    ) throws StatusRuntimeException, InvalidPropertiesFormatException {
        if(allowed_transfer_functions.size() != layer_sizes.size()){
            throw new InvalidPropertiesFormatException("Number of layers doesn't match allowed transfer functions per layer");
        }
        try {
            System.out.print(",,,");
            RafkoDeepLearningService.Build_network_request build_request = RafkoDeepLearningService.Build_network_request.newBuilder()
                    .setTargetSlotId(slot_id).setInputSize(input_size).setExpectedInputRange(expected_input_range)
                    .addAllAllowedTransfersByLayer(allowed_transfer_functions)
                    .addAllLayerSizes(layer_sizes)
                    .build();

            System.out.print("...");
            RafkoDeepLearningService.Slot_response answer = server_rpc.buildNetwork(build_request);

            System.out.println("Built Network in: " + answer.getSlotId());

            return get_network(slot_id);
        } catch (StatusRuntimeException e){
            e.printStackTrace();
            on_disconnect.run();
            throw e;
        }
    }

}
