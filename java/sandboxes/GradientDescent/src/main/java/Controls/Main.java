package Controls;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyCodeCombination;
import javafx.scene.input.KeyCombination;
import javafx.stage.Stage;

import java.io.File;

public class Main extends Application {

    public Stage stage;

    @Override
    public void start(Stage primaryStage) throws Exception{
        FXMLLoader mainLoader = new FXMLLoader(getClass().getClassLoader().getResource("fxml/main.fxml"));

        stage = primaryStage;

        /* Load UI */
        Parent root = mainLoader.load();
        primaryStage.setTitle("Gradient descent? More like Grandiose Depression AMIRIGHT??! ayy...");
        primaryStage.setScene(new Scene(root, 800, 600));
        primaryStage.getScene().getStylesheets().add(getClass().getClassLoader().getResource("css/chart.css").toExternalForm ());

        primaryStage.show();

    }


    public static void main(String[] args) {
        launch(args);
    }
}