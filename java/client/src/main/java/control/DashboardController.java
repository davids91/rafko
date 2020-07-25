package control;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.shape.Rectangle;
import org.rafko.mainframe.RafkoDeepLearningService;
import org.rafko.sparse_net_library.RafkoCommon;
import org.rafko.sparse_net_library.RafkoSparseNet;
import services.RafkoDLClient;

import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.ResourceBundle;

public class DashboardController implements Initializable {
    public Rectangle rect_train_0;
    public Rectangle rect_train_1;
    public Rectangle rect_train_2;
    public Rectangle rect_train_3;
    public Rectangle rect_train_4;
    public Rectangle rect_output_0;
    public Rectangle rect_output_1;
    public Rectangle rect_output_2;
    public Rectangle rect_output_3;
    public Rectangle rect_output_4;
    public TextArea dataset_size_textfield;
    public TextField serverAddress_textField;
    public Label network_folder_label;
    public Label serverStatus_label;
    public Button connect_btn;
    public Button test_btn;
    public Button dataset_load_btn;
    public Button dataset_create_btn;
    public Button network_load_btn;
    public Button network_create_btn;
    public Button save_network_btn;
    public Button gen_sequence_btn;
    public Button start_training_btn;
    public Button play_sequence_btn;
    public Button dataset_save_btn;
    public Slider sample_index_slider;
    public Slider sequence_index_slider;
    public Label server_state_label;
    public ComboBox server_slot_combo;
    public Label server_slot_state_label;
    public Button server_slot_state_query;
    public Button read_network_from_slot_btn;
    public Button create_slot_btn;
    public Button dataset_upload_btn;

    RafkoDLClient client;
    RafkoCommon.Data_set configured_data_set;
    RafkoSparseNet.SparseNet configured_neural_network;
    final int sequence_size = 25;
    final int sample_number = 500;
    boolean server_online;
    boolean network_loaded;
    boolean data_set_available;
    int selected_slot_state;

    public DashboardController(){
        server_online = false;
        network_loaded = false;
        data_set_available = false;
        selected_slot_state = 0;
    }

    void correct_ui_state(){
        ImageView img;

        if(network_loaded){
            save_network_btn.setDisable(false);
        }else{
            save_network_btn.setDisable(true);
        }

        if(data_set_available){
            dataset_save_btn.setDisable(false);
            sample_index_slider.setMax((double)configured_data_set.getInputsCount() / (configured_data_set.getSequenceSize() * configured_data_set.getInputSize()));
            sample_index_slider.setDisable(false);
            sequence_index_slider.setMax(sequence_size);
            sequence_index_slider.setDisable(false);
            play_sequence_btn.setDisable(false);
            if(server_online){
                dataset_upload_btn.setDisable(false);
            }
        }else{
            dataset_upload_btn.setDisable(true);
            dataset_save_btn.setDisable(true);
            sample_index_slider.setMax(0);
            sample_index_slider.setDisable(true);
            sequence_index_slider.setMax(0);
            sequence_index_slider.setDisable(true);
            play_sequence_btn.setDisable(true);
        }

        if(server_online){
            if(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET.ordinal() )) {
                read_network_from_slot_btn.setDisable(false);
            }else{
                read_network_from_slot_btn.setDisable(true);
            }
            if(network_loaded){
                gen_sequence_btn.setDisable(false);
                if(data_set_available  && (selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK.ordinal() ) ){
                    start_training_btn.setDisable(false);
                }else{
                    start_training_btn.setDisable(true);
                }
            }else{
                start_training_btn.setDisable(true);
                gen_sequence_btn.setDisable(true);
            }
            img = new ImageView(new Image("pic/online.png"));
            network_create_btn.setDisable(false);
            server_slot_combo.setDisable(false);
            create_slot_btn.setDisable(false);
            /* Ask the state for the selected Server slot */
            String selected_server_slot_id = (String) server_slot_combo.getSelectionModel().getSelectedItem();
            if(!selected_server_slot_id.equals("")){
                RafkoDeepLearningService.Slot_response slot_state = client.ping();
                String state_string;
                if(0 < (slot_state.getSlotState() & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK.ordinal())){
                    state_string = "A-OK!";
                    network_load_btn.setDisable(false);
                }else{
                    state_string = "Missing: ";
                    if(0 < (slot_state.getSlotState() & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET.ordinal())){
                        state_string += " net ";
                    }else network_load_btn.setDisable(false);
                    if(0 < (slot_state.getSlotState() & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET.ordinal())){
                        state_string += " solution ";
                    }
                    if(0 < (slot_state.getSlotState() & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET.ordinal())){
                        state_string += " data_set ";
                    }
                    if(0 < (slot_state.getSlotState() & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET.ordinal())){
                        state_string += " cost_function ";
                    }
                    if(0 < (slot_state.getSlotState() & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET.ordinal())){
                        state_string += " trainer ";
                    }
                }
                server_slot_state_label.setText(state_string);
            }
        }else{
            img = new ImageView(new Image("pic/offline.png"));
            network_load_btn.setDisable(true);
            network_create_btn.setDisable(true);
            start_training_btn.setDisable(true);
            server_slot_combo.setDisable(true);
            test_btn.setDisable(true);
            create_slot_btn.setDisable(true);
            gen_sequence_btn.setDisable(true);
        }
        img.setPreserveRatio(true);
        img.setFitWidth(64);
        img.setFitHeight(64);
        serverStatus_label.setGraphic(img);
        serverStatus_label.setText("");
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        connect_btn.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {
                    String target = serverAddress_textField.getText();
                    // Create a communication channel to the server, known as a Channel. Channels are thread-safe
                    // and reusable. It is common to create channels at the beginning of your application and reuse
                    // them until the application shuts down.
                    ManagedChannel com_channel = ManagedChannelBuilder.forTarget(target)
                            // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                            // needing certificates.
                            .usePlaintext()
                            .build();
                    client = new RafkoDLClient(com_channel);
                    System.out.println("Client created!");
                    test_btn.setDisable(false);
            }
        });
    }

    @FXML
    void ping_server(){
        if(null != client){
            if(null != client.ping()){
                server_online = true;
            }else{
                server_online = false;
            }
            correct_ui_state();
        }
    }

    @FXML
    void create_network(){
        if(server_online && (selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK.ordinal() ) ){
            configured_neural_network = client.create_network(
                (String)server_slot_combo.getSelectionModel().getSelectedItem(),
                1, 1,
                new ArrayList<RafkoCommon.transfer_functions>(Arrays.asList( /* Allowed transfer functions */
                    RafkoCommon.transfer_functions.TRANSFER_FUNCTION_SELU,
                    RafkoCommon.transfer_functions.TRANSFER_FUNCTION_SELU,
                    RafkoCommon.transfer_functions.TRANSFER_FUNCTION_SELU
                )),
                new ArrayList<Integer>(Arrays.asList(5,10,5)) /* Layer sizes */
            );
            if(null != configured_neural_network){
                network_loaded = true;
            }else{
                network_loaded = false;
            }
            correct_ui_state();
        }
    }

    @FXML
    void create_new_dataset() {
        double sin_value = 0;
        double prev_sin_value = 0;
        Random rnd = new Random();
        double sin_step = Math.PI * 2 / 20;
        ArrayList<Double> sin_inputs = new ArrayList<>(sequence_size * sample_number);
        ArrayList<Double> sin_labels = new ArrayList<>(sequence_size * sample_number);
        for(int sample_index = 0; sample_index < sample_number; ++sample_index){
            sin_value = rnd.nextDouble();
            for(int sequence_index = 0; sequence_index < sequence_size; ++sequence_index){
                prev_sin_value = sin_value;
                sin_value += sin_step;
                sin_inputs.add(Math.sin(prev_sin_value));
                sin_labels.add(Math.sin(sin_value));
            }
        }
        configured_data_set = RafkoCommon.Data_set.newBuilder()
            .setInputSize(5).setFeatureSize(5).setSequenceSize(sequence_size)
            .addAllInputs(sin_inputs).addAllLabels(sin_labels)
            .build();
        if(null != configured_data_set){
            data_set_available = true;
        }else{
            data_set_available = false;
        }
        correct_ui_state();
    }

    @FXML
    void create_server_slot() {
        RafkoDeepLearningService.Service_slot.Builder builder = RafkoDeepLearningService.Service_slot.newBuilder()
            .setType(RafkoDeepLearningService.Slot_type.SERV_SLOT_TO_APPROXIMIZE)
            .setHypers(RafkoDeepLearningService.Service_hyperparameters.newBuilder()
                    .setStepSize(1e-4).setMinibatchSize(64).setMemoryTruncation(3)
                    .build());
        if(network_loaded){
            builder.setNetwork(configured_neural_network);
        }

        if(data_set_available){
            builder.setTrainingSet(configured_data_set);
        }

        RafkoDeepLearningService.Service_slot slot_attempt = builder.build();
    }

    @FXML
    void upload_dataset() {
        if(data_set_available && (selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK.ordinal() ) ){
            RafkoDeepLearningService.Service_slot request = RafkoDeepLearningService.Service_slot.newBuilder()
                .setSlotId((String)server_slot_combo.getSelectionModel().getSelectedItem())
                .setTrainingSet(configured_data_set)
                .build();

            selected_slot_state = client.update_server_slot(request).getSlotState();
            correct_ui_state();
        }
    }
}
