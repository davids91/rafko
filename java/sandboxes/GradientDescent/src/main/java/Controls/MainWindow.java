package Controls;

import Models.ErrorFunction;
import Models.Polynomial;
import javafx.animation.Animation;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollBar;
import javafx.scene.control.Slider;
import javafx.scene.input.ScrollEvent;
import javafx.util.Duration;

import java.util.ArrayList;
import java.util.Random;

public class MainWindow {

    public Button fill_button;
    public LineChart display_graph;
    public Button play_button;
    public Slider speed_slider;
    public Slider dataset_size_slider;
    public Label dataset_size_label;
    public Slider entropy_slider;
    public Button step_button;
    public Label learning_rate_label;
    public Slider learning_rate_slider;
    public LineChart error_graph;

    private Random rnd = new Random();
    private Polynomial solution_trend;
    private Polynomial dataset_trend;
    private ArrayList<Double> dataset;
    private Timeline timeline;

    public void initialize(){
        /* Add slider listeners */
        dataset_size_slider.valueProperty().addListener((observable, oldValue, newValue) -> {
            dataset_size_label.setText(Integer.toString(newValue.intValue()));
        });
        learning_rate_slider.valueProperty().addListener((observable, oldValue, newValue) -> {
                learning_rate_label.setText(String.format("%.10f", newValue));
        });
        timeline = new Timeline(new KeyFrame(Duration.millis(250), ae -> step()));
        fill_chart();
    }

    @FXML
    public  void fill_chart(){
        dataset = new ArrayList((int) dataset_size_slider.getValue());
        display_graph.setData(FXCollections.observableArrayList());
        error_graph.setData(FXCollections.observableArrayList());
        dataset_trend = new Polynomial(rnd,dataset_size_slider.getValue());
        solution_trend = new Polynomial(rnd, dataset_size_slider.getValue());
        XYChart.Series dataset_series = new XYChart.Series();
        dataset_series.setName("Data points");

        for (int i = 0; i < dataset_size_slider.getValue(); ++i) {
            dataset.add(dataset_trend.solve_for(i) + (rnd.nextDouble() - 0.5) * entropy_slider.getValue() * dataset_size_slider.getValue());
            dataset_series.getData().add( new XYChart.Data(i, dataset.get(i)));
        }
        display_graph.getData().add(0, dataset_series);
        display_graph.getData().add(1, new XYChart.Series());
        error_graph.getData().add(0,new XYChart.Series()); /* B error */
        error_graph.getData().add(1,new XYChart.Series()); /* C error */
        displaySolutionTrend();
    }

    @FXML
    public void step() { /* this is where the magic will happen */
        /* Calculate the gradient using every sample in the dataset */
        double gradientA = 0;
        double gradientB = 0;
        double gradientC = 0;
        for (int x = 0; x < dataset.size(); x++) {
            gradientA += -2*x*x*(dataset.get(x) - solution_trend.solve_for(x));
            gradientB += -2*x*(dataset.get(x) - solution_trend.solve_for(x));
            gradientC += -2*(dataset.get(x) - solution_trend.solve_for(x));
        }
        gradientA /= dataset.size();
        gradientB /= dataset.size();
        gradientC /= dataset.size();
        System.out.println("Gradient for A: " +  gradientA);
        System.out.println("Gradient for B: " +  gradientB);
        System.out.println("Gradient for C: " +  gradientC);
        solution_trend.stepA(-gradientA * learning_rate_slider.getValue());
        solution_trend.stepB(-gradientB * learning_rate_slider.getValue());
        solution_trend.stepC(-gradientC * learning_rate_slider.getValue());
        System.out.println("solution_trend.getC()" + solution_trend.getC() + "<>" + dataset_trend.getA());
        System.out.println("solution_trend.getB()" + solution_trend.getB() + "<>" + dataset_trend.getB());
        System.out.println("solution_trend.getC()" + solution_trend.getC() + "<>" + dataset_trend.getC());
        System.out.println("====================");
        displaySolutionTrend();
        if(
            (
                (Math.abs(gradientC) < learning_rate_slider.getValue())
                &&(Math.abs(gradientB) < learning_rate_slider.getValue())
            )&&!(
                (Animation.Status.STOPPED == timeline.getStatus())
                ||(Animation.Status.PAUSED == timeline.getStatus())
            )
        )play_stop();
    }

    @FXML
    public void play_stop() {
        /* this is where the magic will step  */
        if(
            (Animation.Status.STOPPED == timeline.getStatus())
            ||(Animation.Status.PAUSED == timeline.getStatus())
        ){
            play_button.setText("Stop");
            timeline.setCycleCount(Animation.INDEFINITE);
            timeline.play();
        }else{
            play_button.setText("Start");
            timeline.stop();
        }
    }

    public void displaySolutionTrend(){
        XYChart.Series solution_display = new XYChart.Series();
        solution_display.setName("Regression trend");
        solution_display.setData(FXCollections.observableArrayList());
        for (int i = 0; i < dataset_size_slider.getValue(); ++i) {
            solution_display.getData().add( new XYChart.Data(i, solution_trend.solve_for(i)));
        }
        display_graph.getData().set(1,solution_display);
    }
}
