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
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.stage.Stage;
import services.RafkoDLClient;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

public class DashboardController implements Initializable {
    public TextField serverAddress_textField;
    public Button connect_btn;
    public Label serverStatus_label;
    public Button test_btn;

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
}
