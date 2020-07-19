package control;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.shape.Rectangle;
import javafx.stage.Stage;
import services.RafkoDLClient;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.net.URL;
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

    RafkoDLClient client;

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
            ImageView img;
            if(client.ping()){
                img = new ImageView(new Image("pic/online.png"));
            }else{
                img = new ImageView(new Image("pic/offline.png"));
                test_btn.setDisable(true);
            }
            img.setPreserveRatio(true);
            img.setFitWidth(64);
            img.setFitHeight(64);
            serverStatus_label.setGraphic(img);
            serverStatus_label.setText("");
        }
    }

    @FXML
    void create_network(){

    }
}
