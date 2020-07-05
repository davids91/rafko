package services;

import io.grpc.Channel;
import io.grpc.StatusRuntimeException;
import org.rafko.mainframe.RafkoDeepLearningService;
import org.rafko.mainframe.Rafko_deep_learningGrpc;
import rafko_mainframe.DeepLearningService;

import java.util.logging.Level;
import java.util.logging.Logger;

public class RafkoDLClient {
    private static final Logger logger = Logger.getLogger(services.RafkoDLClient.class.getName());
    private Rafko_deep_learningGrpc.Rafko_deep_learningBlockingStub server_stub;

    public RafkoDLClient(Channel channel){
        server_stub = Rafko_deep_learningGrpc.newBlockingStub(channel);
    }

    public boolean ping(){
        /* if any commend is successful, then server is online! */
        RafkoDeepLearningService.Slot_request request = RafkoDeepLearningService.Slot_request
            .newBuilder().setTargetSlotId("MOOT").setAction(RafkoDeepLearningService.Slot_action.SERV_SLOT_NOOP).build();
        try {
            org.rafko.sparse_net_library.RafkoSparseNet.SparseNet dummy = server_stub.getNetwork(request);
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return false;
        } catch(Exception e){
            e.printStackTrace();
            return false;
        }
        return true;
    }

}