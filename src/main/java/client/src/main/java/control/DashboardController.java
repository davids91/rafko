package control;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import javafx.animation.Animation;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.shape.Rectangle;
import javafx.stage.FileChooser;
import javafx.util.Duration;
import javafx.util.StringConverter;
import models.ColorMap;
import models.Global;
import models.ServerSlot_data;
import org.rafko.mainframe.RafkoDeepLearningService;
import org.rafko.rafko_net.RafkoCommon;
import org.rafko.rafko_net.RafkoSparseNet;
import org.rafko.rafko_net.RafkoTraining;
import services.RafkoDLClient;

import java.io.*;
import java.net.URL;
import java.sql.Time;
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
    public TextField serverAddress_textField;
    public Label network_folder_label;
    public Button create_client_btn;
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
    public ComboBox<ServerSlot_data> server_slot_combo;
    public Label server_slot_state_label;
    public Button server_slot_state_query;
    public Button create_slot_btn;
    public Button load_slot_btn;
    public Tooltip slot_state_tooltip;
    public Label sample_index_label;
    public LineChart<Integer,Double> error_chart;

    RafkoDLClient client;
    RafkoSparseNet.SparseNet loaded_neural_network;
    final int prefill_size = 2;
    final int sequence_size = 5;
    final int sample_number = 50;
    boolean server_online;
    boolean training_started;
    int selected_slot_state;
    FileChooser network_filechooser;
    XYChart.Series<Integer, Double> error_series;
    Timeline chart_timeline;
    Timeline chart_supervisor_timeline;
    double error_moving_average;
    double error_moving_average_last_value;

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        server_online = false;
        training_started = false;
        selected_slot_state = 0;
        network_filechooser = new FileChooser();
        network_filechooser.getExtensionFilters().add(
            new FileChooser.ExtensionFilter("Rafko Neural Network files (*.rnn)", "*.rnn")
        );
        network_filechooser.setInitialDirectory(new File("D:/casdev/temp"));
        error_chart.setTitle("Training errors / iteration");
        error_series = new XYChart.Series<>();
        error_chart.getData().add(error_series);
        chart_timeline = new Timeline(new KeyFrame(Duration.millis(500), event -> ask_for_progress()));
        chart_timeline.setCycleCount(Animation.INDEFINITE);
        chart_supervisor_timeline = new Timeline(new KeyFrame(Duration.millis(1000), event -> evaluate_chart_timings()));
        chart_supervisor_timeline.setCycleCount(Animation.INDEFINITE);
        error_moving_average = 0;
        error_moving_average_last_value = 0;

        create_client_btn.setOnAction(event -> {
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
                server_slot_combo.getItems().clear();
                server_online = false;
                server_slot_combo.getItems().clear();
                correct_ui_state();
            });
            test_btn.setDisable(false);
            ping_server();
        });
        server_slot_combo.setConverter(new StringConverter<>(){
            @Override
            public String toString(ServerSlot_data object) {
                if(null == object) return null;
                else return object.getName();
            }

            @Override
            public ServerSlot_data fromString(String string) {
                return new ServerSlot_data(string);
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
            correct_ui_state();
        });
    }

    void evaluate_chart_timings(){
        double next_interval = Math.max(300,Math.min(7200,
            1000 / (Math.max(0.01, Math.abs(error_moving_average - error_moving_average_last_value)))
        ));// * 1000; /* convert to seconds */
        chart_timeline.stop();
        chart_timeline = new Timeline((new KeyFrame(
            Duration.millis(next_interval), event_ -> ask_for_progress())
        ));
        error_moving_average_last_value = error_moving_average;
        chart_supervisor_timeline.stop();
        chart_supervisor_timeline = new Timeline(new KeyFrame(Duration.millis(next_interval / 10.0), event -> evaluate_chart_timings()));
        chart_supervisor_timeline.setCycleCount(Animation.INDEFINITE);
        if(training_started){
            chart_supervisor_timeline.play();
            chart_timeline.play();
        }
    }

    void ask_for_progress(){
        if(
            (0 < selected_slot_state)
            &&(
                (selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
                ||(
                    (0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE))
                    &&(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_TRAINER_VALUE))
                )
            )
        ){
            RafkoDeepLearningService.Slot_info progress = client.get_info(
                RafkoDeepLearningService.Slot_request.newBuilder()
                .setTargetSlotId(server_slot_combo.getValue().getName())
                .setRequestBitstring(
                        RafkoDeepLearningService.Slot_info_field.SLOT_INFO_ITERATION_VALUE
                        | RafkoDeepLearningService.Slot_info_field.SLOT_INFO_TRAINING_ERROR_VALUE
                )
                .build()
            );
            error_moving_average = (error_moving_average + progress.getInfoPackage(1)) / 2.0;
            error_series.getData().add(new XYChart.Data<>(
                (int)progress.getInfoPackage(0), /* Iteration */
                progress.getInfoPackage(1)) /* Training error */
            );
        }
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

        display_output(sample);
    }

    void display_output(RafkoDeepLearningService.Neural_io_stream sample){
        /* Display the data from the end of the package */
        rect_output_3.setFill(ColorMap.getColor(sample.getPackage(sample.getPackageCount() - 5)));
        rect_output_4.setFill(ColorMap.getColor(sample.getPackage(sample.getPackageCount() - 4)));
        rect_output_5.setFill(ColorMap.getColor(sample.getPackage(sample.getPackageCount() - 3)));
        rect_output_6.setFill(ColorMap.getColor(sample.getPackage(sample.getPackageCount() - 2)));
        rect_output_7.setFill(ColorMap.getColor(sample.getPackage(sample.getPackageCount() - 1)));
    }

    void correct_ui_state(){
        ImageView img;
        if(server_online){
            server_slot_combo.setDisable(false);
            if(null != server_slot_combo.getValue()){
                server_slot_state_query.setDisable(false);
                selected_slot_state = client.ping(server_slot_combo.getValue().getName()).getSlotState();
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

                RafkoDeepLearningService.Slot_info slot_info = client.get_info(RafkoDeepLearningService.Slot_request.newBuilder()
                        .setTargetSlotId(server_slot_combo.getValue().getName())
                        .setRequestBitstring(RafkoDeepLearningService.Slot_info_field.SLOT_INFO_TRAINING_SET_SEQUENCE_COUNT_VALUE)
                        .build());
                if(training_started){
                    start_training_btn.setText("Stop Training");
                }else{
                    start_training_btn.setText("Start Training");
                }
                if(
                    (0 < selected_slot_state)
                    &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
                    ||(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_DATA_SET_VALUE)))
                    &&(1 == slot_info.getInfoPackageCount())
                ){
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

            }else{
                server_slot_state_query.setDisable(true);
                dataset_load_btn.setDisable(true);
                dataset_create_btn.setDisable(true);
                dataset_save_btn.setDisable(true);
                sample_index_slider.setMax(0);
                sample_index_slider.setDisable(true);
                network_folder_label.setText("<No Network Loaded>");
                save_network_btn.setDisable(true);
                chart_timeline.stop();
            }
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
                    chart_timeline.stop();
                    start_training_btn.setDisable(true);
                    error_series.getData().clear();
                }
            }else{
                start_training_btn.setDisable(true);
                gen_sequence_btn.setDisable(true);
            }
            img = new ImageView(new Image("pic/online.png"));
            network_create_btn.setDisable(false);
            server_slot_combo.setDisable(false);
            create_slot_btn.setDisable(false);
            //load_slot_btn.setDisable(false);
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
            load_slot_btn.setDisable(true);
            gen_sequence_btn.setDisable(true);
            server_slot_combo.getItems().clear();
            server_slot_combo.setDisable(true);
            chart_timeline.stop();
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
            display_output(client.run_net_once(server_slot_combo.getValue().getName(),(int)sample_index_slider.getValue()));
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
                        RafkoCommon.transfer_functions.transfer_function_selu,
                        RafkoCommon.transfer_functions.transfer_function_selu,
                        RafkoCommon.transfer_functions.transfer_function_selu
                    )),
                    new ArrayList<>(Arrays.asList(4,2,1)) /* Layer sizes */
                );

                if(0 < loaded_neural_network.getNeuronArrayCount()){
                    selected_slot_state = client.ping(server_slot_combo.getValue().getName()).getSlotState();
                    correct_ui_state();
                    return;
                }
            } catch (InvalidPropertiesFormatException e) {
                e.printStackTrace();
                loaded_neural_network = null;
            }
        }
        selected_slot_state = client.ping(server_slot_combo.getValue().getName()).getSlotState();
        correct_ui_state();
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
        upload_dataset(RafkoTraining.DataSet.newBuilder()
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
        }else {
            correct_ui_state();
        }
    }

    void upload_dataset(RafkoTraining.DataSet data_set) {
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
            correct_ui_state();
        }
    }

    void load_network_from_slot(){
        if(
            (0 < selected_slot_state)
            &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
            ||(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE)))
        ){
            loaded_neural_network = client.get_network(server_slot_combo.getValue().getName());
            if((null != loaded_neural_network)&&(0 < loaded_neural_network.getNeuronArrayCount())){
                correct_ui_state();
                return;
            }
        }
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
        correct_ui_state();
    }

    @FXML
    void save_network_to_fs(){
        if(
            (0 < selected_slot_state)
            &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
            ||(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE)))
        ){
            File network_file = network_filechooser.showSaveDialog(Global.primaryStage);
            try (FileOutputStream stream = new FileOutputStream(network_file)){
                stream.write(loaded_neural_network.toByteArray());
            } catch (IOException e) {
                new Alert(Alert.AlertType.ERROR,"File Save not successful").showAndWait();
                e.printStackTrace();
            }
        }else{
            correct_ui_state();
        }
    }

    @FXML
    void lead_serve_slot(){

    }

    @FXML
    void create_server_slot() {
        RafkoDeepLearningService.Service_slot.Builder builder = RafkoDeepLearningService.Service_slot.newBuilder()
            .setType(RafkoDeepLearningService.Slot_type.SERV_SLOT_TO_APPROXIMIZE)
            .setHypers(RafkoDeepLearningService.Service_hyperparameters.newBuilder()
                    .setAlpha(1.6732).setLambda(1.0507)
                    .setStepSize(1e-2).setMinibatchSize(64).setMemoryTruncation(3)
                    .build())
            .setCostFunction(RafkoCommon.cost_functions.cost_function_mse)
            .setWeightUpdater(RafkoCommon.weight_updaters.weight_updater_momentum);

        if(
            (0 < selected_slot_state)
            &&((selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
            ||(0 == (selected_slot_state & RafkoDeepLearningService.Slot_state_values.SERV_SLOT_MISSING_NET_VALUE)))
        ){
            builder.setNetwork(loaded_neural_network);
        }

        RafkoDeepLearningService.Service_slot slot_attempt = builder.build();
        RafkoDeepLearningService.Slot_response slot_state = client.add_server_slot(slot_attempt);

        if(
            (0 < slot_state.getSlotState())
            &&(!slot_state.getSlotId().isEmpty())
        ){
            /* server_slot_combo.add */
            server_slot_combo.getItems().add(new ServerSlot_data(slot_state.getSlotId()));
            server_slot_combo.getSelectionModel().selectLast();
            create_new_dataset();
            create_network();
            correct_ui_state();
            sample_index_slider.setValue(0);
        }else ping_server();
    }

    @FXML
    void query_slot_sate(){
        if(server_online && (null != client) && (null != server_slot_combo.getValue().getName())){
            selected_slot_state = client.ping(server_slot_combo.getValue().getName()).getSlotState();
            correct_ui_state();
        }else selected_slot_state = 0;
    }

    void start_training(){
        if(
            (0 < selected_slot_state)
            &&(selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
        ){
            client.request_one_action(
                server_slot_combo.getValue().getName(),
                RafkoDeepLearningService.Slot_action_field.SERV_SLOT_TO_START_VALUE,0
            );
            System.out.println("Action done!");
            chart_timeline.play();
            chart_supervisor_timeline.play();
            training_started = true;
        }else correct_ui_state();
    }

    void stop_training(){
        if(
            (0 < selected_slot_state)
            &&(selected_slot_state == RafkoDeepLearningService.Slot_state_values.SERV_SLOT_OK_VALUE)
        ){
            client.request_one_action(
                server_slot_combo.getValue().getName(),
                RafkoDeepLearningService.Slot_action_field.SERV_SLOT_TO_STOP_VALUE,0
            );
            chart_timeline.stop();
            chart_supervisor_timeline.stop();
            training_started = false;
        }else correct_ui_state();
    }

    @FXML
    void training_start_stop(){
        if(!training_started){
            start_training();
        }else{
            stop_training();
        }
        correct_ui_state();
    }
}
