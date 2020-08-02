package control;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.shape.Rectangle;
import javafx.stage.FileChooser;
import javafx.util.StringConverter;
import models.ColorMap;
import models.Global;
import models.Server_slot_data;
import org.rafko.mainframe.RafkoDeepLearningService;
import org.rafko.mainframe.Rafko_deep_learningGrpc;
import org.rafko.sparse_net_library.RafkoCommon;
import org.rafko.sparse_net_library.RafkoSparseNet;
import services.RafkoDLClient;

import java.io.*;
import java.net.URL;
import java.util.*;

public class DashboardController implements Initializable {
    public Rectangle rect_input_1;
    public Rectangle rect_input_2;
    public Rectangle rect_input_3;
    public Rectangle rect_input_4;
    public Rectangle rect_input_5;
    public Rectangle rect_input_6;
    public Rectangle rect_input_7;
    public Rectangle rect_output_1;
    public Rectangle rect_output_2;
    public Rectangle rect_output_3;
    public Rectangle rect_output_4;
    public Rectangle rect_output_5;
    public Rectangle rect_output_6;
    public Rectangle rect_output_7;
    public TextArea dataset_size_textfield;
    public TextField serverAddress_textField;
    public Label network_folder_label;
    public Button connect_btn;
    public Button test_btn;
    public Button dataset_load_btn;
    public Button dataset_create_btn;
    public Button network_load_btn;
    public Button network_create_btn;
    public Button save_network_btn;
    public Button gen_sequence_btn;
    public Button start_training_btn;
    public Button dataset_save_btn;
    public Slider sample_index_slider;
    public Label server_state_label;
    public ComboBox<Server_slot_data> server_slot_combo;
    public Label server_slot_state_label;
    public Button server_slot_state_query;
    public Button create_slot_btn;
    public Tooltip slot_state_tooltip;
    public Label sample_index_label;

    RafkoDLClient client;
    RafkoSparseNet.SparseNet loaded_neural_network;
    final int prefill_size = 2;
    final int sequence_size = 5;
    final int sample_number = 500;
    boolean server_online;
    int selected_slot_state;
    FileChooser network_filechooser;
    RafkoDeepLearningService.Neural_io_stream selected_sample;

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        server_online = false;
        selected_slot_state = 0;
        network_filechooser = new FileChooser();
        network_filechooser.getExtensionFilters().add(
            new FileChooser.ExtensionFilter("Rafko Neural Network files (*.rnn)", "*.rnn")
        );
        network_filechooser.setInitialDirectory(new File("D:/casdev/temp"));

        connect_btn.setOnAction(event -> {
            String target = serverAddress_textField.getText();
            // Create a communication channel to the server, known as a Channel. Channels are thread-safe
            // and reusable. It is common to create channels at the beginning of your application and reuse
            // them until the application shuts down.
            ManagedChannel com_channel = ManagedChannelBuilder.forTarget(target)
                    // Channels are secure by default (via SSL/TLS). For the example we disable TLS to avoid
                    // needing certificates.
                    .usePlaintext()
                    .build();
            client = new RafkoDLClient(com_channel, () -> {
                server_online = false;
                server_slot_combo.getItems().clear();
                System.out.println("on disconnect..");
                correct_ui_state();
            });
            System.out.println("Client created!");
            test_btn.setDisable(false);
            ping_server();
        });
        server_slot_combo.setConverter(new StringConverter<>(){
            @Override
            public String toString(Server_slot_data object) {
                if(null == object) return null;
                else return object.getName();
            }

            @Override
            public Server_slot_data fromString(String string) {
                return new Server_slot_data(string);
            }
        });
        server_slot_combo.getSelectionModel().selectedItemProperty().addListener((observable, oldValue, newValue) -> {
            selected_slot_state = client.ping(newValue.getName()).getSlotState();
            if(
                (0 < selected_slot_state)
                &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
                ||(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE)))
            ){
                load_network_from_slot();
            }
            System.out.println("on combobox selection changed..");
            correct_ui_state();
        });

        sample_index_slider.valueProperty().addListener((observable, oldValue, newValue) -> {
            sample_index_label.setText((int)sample_index_slider.getValue() + "/" + (int)sample_index_slider.getMax());
            if(
                    (0 < selected_slot_state)
                    &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
                    ||(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_DATA_SET_VALUE)))
            ){
                /* Set data from solution  */
                display_sample(client.request_one_action(
                    server_slot_combo.getValue().getName(),
                    RafkoDeepLearningService.Slot_action_field.SERV_SLOT_TO_GET_TRAINING_SAMPLE_VALUE,
                    newValue.intValue()
                ).getDataStream());
            }
            System.out.println("on slider change..");
            correct_ui_state();
        });
    }

    void display_sample(RafkoDeepLearningService.Neural_io_stream sample){
        /* Read in the input(1 input for 5 sequence) */
        rect_input_1.setFill(ColorMap.getColor(sample.getPackage(0)));
        rect_input_2.setFill(ColorMap.getColor(sample.getPackage(1)));
        rect_input_3.setFill(ColorMap.getColor(sample.getPackage(2)));
        rect_input_4.setFill(ColorMap.getColor(sample.getPackage(3)));
        rect_input_5.setFill(ColorMap.getColor(sample.getPackage(4)));
        rect_input_6.setFill(ColorMap.getColor(sample.getPackage(5)));
        rect_input_7.setFill(ColorMap.getColor(sample.getPackage(6)));

        /* Read in the output */
        rect_output_3.setFill(ColorMap.getColor(sample.getPackage(7)));
        rect_output_4.setFill(ColorMap.getColor(sample.getPackage(8)));
        rect_output_5.setFill(ColorMap.getColor(sample.getPackage(9)));
        rect_output_6.setFill(ColorMap.getColor(sample.getPackage(10)));
        rect_output_7.setFill(ColorMap.getColor(sample.getPackage(11)));
    }

    void correct_ui_state(){
        ImageView img;
        System.out.println("Correct ui state..");
        if(
            (0 < selected_slot_state)
            &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
            ||(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE)))
        ){
            network_folder_label.setText(decide_network_name());
            save_network_btn.setDisable(false);
        }else{
            network_folder_label.setText("<No Network Loaded>");
            save_network_btn.setDisable(true);
        }

        if(
            (0 < selected_slot_state)
            &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
            ||(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_DATA_SET_VALUE)))
        ){
            RafkoDeepLearningService.Slot_info slot_info = client.get_info(RafkoDeepLearningService.Slot_request.newBuilder()
                .setTargetSlotId(server_slot_combo.getValue().getName())
                .setRequestBitstring(RafkoDeepLearningService.Slot_info_field.SLOT_INFO_TRAINING_SET_SEQUENCE_COUNT.ordinal())
                .build());
            dataset_save_btn.setDisable(false);
            sample_index_slider.setMax(Math.max(1,slot_info.getInfoPackage(    0)-1));
            sample_index_slider.setDisable(false);
            dataset_load_btn.setDisable(false);
            dataset_create_btn.setDisable(false);
        }else{
            dataset_load_btn.setDisable(true);
            dataset_create_btn.setDisable(true);
            dataset_save_btn.setDisable(true);
            sample_index_slider.setMax(0);
            sample_index_slider.setDisable(true);
        }

        if(server_online){
            System.out.println("Server is online!");
            server_slot_combo.setDisable(false);
            if(null != server_slot_combo.getValue()){
                server_slot_state_query.setDisable(false);
                selected_slot_state = client.ping(server_slot_combo.getValue().getName()).getSlotState();
            }else{
                server_slot_state_query.setDisable(true);
            }
            System.out.println("Selected slot state:" + selected_slot_state);
            if(
                (0 < selected_slot_state)
                &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
                ||(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE)))
            ){
                if(
                    (selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
                    ||  (0 == (selected_slot_state &
                    (RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE
                            |RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_SOLUTION_VALUE
                            |RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_DATA_SET_VALUE)
                    ))
                ) {
                    gen_sequence_btn.setDisable(false);
                }else{
                    gen_sequence_btn.setDisable(true);
                }
                if(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_TRAINER_VALUE )) {
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
            String selected_server_slot_id = "";
            if(null != server_slot_combo.getValue()) selected_server_slot_id = server_slot_combo.getValue().getName();
            if((null != selected_server_slot_id) && !selected_server_slot_id.equals("")){
                String state_string;
                if(0 == selected_slot_state){
                    state_string = "==UNKNOWN==";
                    network_load_btn.setDisable(true);
                }else if(0 < (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)){
                    state_string = "A-OK!";
                    network_load_btn.setDisable(false);
                }else{
                    network_load_btn.setDisable(false);
                    state_string = "Missing: ";
                    if(0 < (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE)){
                        state_string += " net ";
                    }
                    if(0 < (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_SOLUTION_VALUE)){
                        state_string += " solution ";
                    }
                    if(0 < (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_DATA_SET_VALUE)){
                        state_string += " data_set ";
                    }
                    if(0 < (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_COST_FUNCTION_VALUE)){
                        state_string += " cost_function ";
                    }
                    if(0 < (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_TRAINER_VALUE)){
                        state_string += " trainer ";
                    }
                }
                server_slot_state_label.setText(state_string);
                slot_state_tooltip.setText(state_string);
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
            server_slot_combo.setDisable(true);
        }
        img.setPreserveRatio(true);
        img.setFitWidth(32);
        img.setFitHeight(32);
        server_state_label.setGraphic(img);
        server_state_label.setText("");
    }

    @FXML
    void ping_server(){
        if(null != client){
            if(null != client.ping()){
                server_online = true;
            }else{
                server_online = false;
            }
            System.out.println("on ping..");
            correct_ui_state();
        }
    }

    String decide_network_name(){
        if(null != server_slot_combo.getValue())
            return decide_network_name(server_slot_combo.getValue().getName());
        else return "";
    }
    String decide_network_name(String source_name){
        String net_name = source_name;
        String consonants = "bcdfghjklmnopqrstvwxyz";
        StringBuilder result_name = new StringBuilder();
        int add_filter = 0;
        net_name = net_name.replaceAll("[^A-Za-z]",""); /* Replace numberic characters */

        for(int i = 0; i < net_name.length(); ++i){
            if(consonants.contains(net_name.substring(i, i+1))){
                if(0 == (add_filter % 3))
                    result_name.append(net_name, i, i + 1);
                ++add_filter;
            }else result_name.append(net_name, i, i + 1);
        }
        return result_name.toString();
    }

    @FXML
    void generate_from_selected_inputs(){
        if(
            (0 < selected_slot_state)
            &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
            ||  (0 == (selected_slot_state &
                    (RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE
                    |RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_SOLUTION_VALUE
                    |RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_DATA_SET_VALUE)
                ))
            )
        ){
            client.run_net_once(server_slot_combo.getValue().getName(),(int)sample_index_slider.getValue());
        }
    }

    @FXML
    void create_network(){
        System.out.print("Selected slot state:" + selected_slot_state);
        if(server_online && (selected_slot_state != RafkoDeepLearningService.Slot_state_values.SERV_SLOT_STATE_UNKNOWN_VALUE ) ){
            System.out.print("creating network..");
            try {
                loaded_neural_network = client.create_network(
                    server_slot_combo.getValue().getName(),
                    1, 1,
                    new ArrayList<>(Arrays.asList( /* Allowed transfer functions */
                        RafkoCommon.transfer_functions.TRANSFER_FUNCTION_SELU,
                        RafkoCommon.transfer_functions.TRANSFER_FUNCTION_SELU,
                        RafkoCommon.transfer_functions.TRANSFER_FUNCTION_SELU
                    )),
                    new ArrayList<>(Arrays.asList(5,10,5)) /* Layer sizes */
                );

                if(0 < loaded_neural_network.getNeuronArrayCount()){
                    selected_slot_state = client.ping(server_slot_combo.getValue().getName()).getSlotState();
                    System.out.println("on create net success..");
                    correct_ui_state();
                    return;
                }
            } catch (InvalidPropertiesFormatException e) {
                e.printStackTrace();
                loaded_neural_network = null;
            }
        }
        selected_slot_state = client.ping(server_slot_combo.getValue().getName()).getSlotState();
        System.out.println("on create net fail..");
        correct_ui_state();
    }

    @FXML
    void create_new_dataset() {
        System.out.println("creating dataset..");
        double sin_value = 0;
        double prev_sin_value = 0;
        Random rnd = new Random();
        double sin_step = Math.PI * 2 / 20;
        ArrayList<Double> sin_inputs = new ArrayList<>(sequence_size * sample_number);
        ArrayList<Double> sin_labels = new ArrayList<>(sequence_size * sample_number);
        for(int sample_index = 0; sample_index < sample_number; ++sample_index){
            sin_value = rnd.nextDouble();
            for(int prefill_index = 0; prefill_index < prefill_size; ++prefill_index){
                prev_sin_value = sin_value;
                sin_value += sin_step;
                sin_inputs.add(Math.sin(prev_sin_value));
            }
            for(int sequence_index = 0; sequence_index < sequence_size; ++sequence_index){
                prev_sin_value = sin_value;
                sin_value += sin_step;
                sin_inputs.add(Math.sin(prev_sin_value));
                sin_labels.add(Math.sin(sin_value));
            }
        }

        System.out.println("Inputs size: " + sin_inputs.size());
        System.out.println("Labels size: " + sin_labels.size());

        upload_dataset(RafkoCommon.Data_set.newBuilder()
            .setInputSize(1).setFeatureSize(1).setSequenceSize(sequence_size)
            .addAllInputs(sin_inputs).addAllLabels(sin_labels)
            .build()
        );
        load_dataset_sample_from_slot((int)sample_index_slider.getValue());
    }

    void load_dataset_sample_from_slot(int sample_index){
        if(
            (0 < selected_slot_state)
            &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
            ||(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_DATA_SET_VALUE)))
        ){
            System.out.println("on load dataset to be implemented..");
        }else {
            System.out.println("on load dataset fail..");
            correct_ui_state();
        }
    }

    void upload_dataset(RafkoCommon.Data_set data_set) {
        System.out.println("on creating dataset..");
        if(
            (null != data_set)
            &&(0 < selected_slot_state)
        ){
            RafkoDeepLearningService.Service_slot request = RafkoDeepLearningService.Service_slot.newBuilder()
                    .setSlotId(server_slot_combo.getValue().getName())
                    .setType(RafkoDeepLearningService.Slot_type.SERV_SLOT_TO_APPROXIMIZE)
                    .setTrainingSet(data_set)
                    .build();

            selected_slot_state = client.update_server_slot(request).getSlotState();
            System.out.println("on upload dataset..");
            correct_ui_state();
        }
        System.out.println("on failed upload dataset..");
    }

    void load_network_from_slot(){
        if(
            (0 < selected_slot_state)
            &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
            ||(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE)))
        ){
            loaded_neural_network = client.get_network(server_slot_combo.getValue().getName());
            if((null != loaded_neural_network)&&(0 < loaded_neural_network.getNeuronArrayCount())){
                System.out.println("on load network success..");
                correct_ui_state();
                return;
            }
        }
        System.out.println("on load network fail..");
        correct_ui_state();
    }

    @FXML
    void load_network_from_fs(){
        if(server_online && (0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE))){
            File network_file = network_filechooser.showSaveDialog(Global.primaryStage);
            try (FileInputStream stream = new FileInputStream(network_file)){
                if(network_file.exists()){
                    loaded_neural_network = RafkoSparseNet.SparseNet.parseFrom(stream.readAllBytes());
                    if(server_online && (selected_slot_state != 0)){
                        client.update_server_slot(
                            RafkoDeepLearningService.Service_slot.newBuilder()
                                .setNetwork(loaded_neural_network)
                                .build()
                        );
                        System.out.println("on read network success..");
                        correct_ui_state();
                    }
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
                new Alert(Alert.AlertType.ERROR,"File not found!").showAndWait();
            } catch (IOException e) {
                e.printStackTrace();
                new Alert(Alert.AlertType.ERROR,"File unavailable").showAndWait();
            }
        }
        System.out.println("on read network fail..");
        correct_ui_state();
    }

    @FXML
    void save_network_to_fs(){
        if(
            (0 < selected_slot_state)
            &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
            ||(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE)))
        ){
            File network_file = network_filechooser.showOpenDialog(Global.primaryStage);
            try (FileOutputStream stream = new FileOutputStream(network_file)){
                stream.write(loaded_neural_network.toByteArray());
            } catch (IOException e) {
                new Alert(Alert.AlertType.ERROR,"File Save not successful").showAndWait();
                e.printStackTrace();
            }
        }else{
            System.out.println("on save network failed..");
            correct_ui_state();
        }
    }

    @FXML
    void create_server_slot() {
        System.out.println("Creating server slot.");
        RafkoDeepLearningService.Service_slot.Builder builder = RafkoDeepLearningService.Service_slot.newBuilder()
            .setType(RafkoDeepLearningService.Slot_type.SERV_SLOT_TO_APPROXIMIZE)
            .setHypers(RafkoDeepLearningService.Service_hyperparameters.newBuilder()
                    .setStepSize(1e-4).setMinibatchSize(64).setMemoryTruncation(3)
                    .build())
            .setCostFunction(RafkoCommon.cost_functions.COST_FUNCTION_MSE)
            .setWeightUpdater(RafkoCommon.weight_updaters.WEIGHT_UPDATER_MOMENTUM);

        if(
            (0 < selected_slot_state)
            &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
            ||(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE)))
        ){
            builder.setNetwork(loaded_neural_network);
            System.out.println("network is valid.");
        }

        RafkoDeepLearningService.Service_slot slot_attempt = builder.build();
        RafkoDeepLearningService.Slot_response slot_state = client.add_server_slot(slot_attempt);
        System.out.println("finished trying to add slot.");

        if(
            (0 < slot_state.getSlotState())
            &&(!slot_state.getSlotId().isEmpty())
        ){
            System.out.println("slot created.");
            /* server_slot_combo.add */
            server_slot_combo.getItems().add(new Server_slot_data(slot_state.getSlotId()));
            server_slot_combo.getSelectionModel().selectLast();
            create_new_dataset();
            create_network();
            sample_index_slider.setValue(0);
        }else ping_server();

    }

    @FXML
    void query_slot_sate(){
        if(server_online && (null != client) && (null != server_slot_combo.getValue().getName())){
            selected_slot_state = client.ping(server_slot_combo.getValue().getName()).getSlotState();
            System.out.println("on query slot state..");
            correct_ui_state();
        }else selected_slot_state = 0;
    }
}
