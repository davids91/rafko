package control;

import io.grpc.ManagedChannelBuilder;
import io.grpc.Server;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.net.URL;
import java.util.ResourceBundle;

public class DashboardEntryPoint extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        FXMLLoader loader = new FXMLLoader();
        Parent root = loader.load(getClass().getResource("../fxml/main.fxml").openStream());
        DashboardController controller = loader.getController();
        primaryStage.setTitle("Rafko Deep Learning Client Dashboard");
        primaryStage.setScene(new Scene(root,800,600));
        primaryStage.show();
    }
    public static void main(String[] args){ launch(args); }
}
