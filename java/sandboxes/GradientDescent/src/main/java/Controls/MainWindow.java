package Controls;

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

    private Random rnd = new Random();
    private Polynomial previous_trend;
    private Polynomial solution_trend;
    private ArrayList<Double> dataset;
    private double previous_error;
    private Timeline timeline;

    public void initialize(){
        /* Add slider listeners */
        dataset_size_slider.valueProperty().addListener((observable, oldValue, newValue) -> {
            dataset_size_label.setText(Integer.toString(newValue.intValue()));
        });
        learning_rate_slider.valueProperty().addListener((observable, oldValue, newValue) -> {
            learning_rate_label.setText(String.format("%.3f", newValue));
        });
        timeline = new Timeline(new KeyFrame(Duration.millis(250), ae -> step()));
        fill_chart();
    }

    @FXML
    public  void fill_chart(){
        dataset = new ArrayList((int) dataset_size_slider.getValue());
        display_graph.setData(FXCollections.observableArrayList());
        Polynomial dataset_trend = new Polynomial(rnd,dataset_size_slider.getValue());
        previous_trend = new Polynomial(rnd,dataset_size_slider.getValue());
        solution_trend = new Polynomial(previous_trend, 1.1);
        previous_error = 0;
        XYChart.Series dataset_series = new XYChart.Series();
        XYChart.Series regression_line = new XYChart.Series();
        dataset_series.setName("Data points");
        regression_line.setName("Regression trend");
        regression_line.getData().add( new XYChart.Data(0,previous_trend.solve_for(0)));
        for (int i = 0; i < dataset_size_slider.getValue(); ++i) {
            dataset.add(dataset_trend.solve_for(i) + (rnd.nextDouble() - 0.5) * entropy_slider.getValue() * dataset_size_slider.getValue());
            dataset_series.getData().add( new XYChart.Data(i, dataset.get(i)));
            regression_line.getData().add( new XYChart.Data(i,previous_trend.solve_for(i)));
            previous_error += Math.pow(dataset.get(i) - previous_trend.solve_for(i),2); /* Squared loss */
        }
        previous_error /= dataset_size_slider.getValue();
        display_graph.getData().add(0, dataset_series);
        display_graph.getData().add(1, regression_line);
    }

    @FXML
    public void step() { /* this is where the magic will happen */
        /* Calculate sum of solution difference from dataset */
        double error = 0;
        double improvement_rate;
        for (int i = 0; i < dataset_size_slider.getValue(); ++i) {
            error += Math.pow(dataset.get(i) - solution_trend.solve_for(i),2); /* Squared loss */
        }
        error /= dataset_size_slider.getValue();
        improvement_rate = Math.max(0.0,Math.min(1.0, error/previous_error));
        /* Calculate the step size and step */
        double step_size =
            (previous_error - error) / (solution_trend.distance(previous_trend))
            * learning_rate_slider.getValue();
        System.out.println("previous_error:" + previous_error);
        System.out.println("previous_trend.getC():" + previous_trend.getC());
        System.out.println("----");
        System.out.println("error:" + error);
        System.out.println("solution_trend.getC():" + solution_trend.getC());
        System.out.println("----");
        System.out.println("(error - previous_error):" + (error - previous_error));
        System.out.println("gradient:" + (previous_error - error) / (solution_trend.distance(previous_trend)));
        System.out.println("(solution_trend.getC() - previous_trend.getC()):" + (solution_trend.getC() - previous_trend.getC()));
        System.out.println("step size:" + step_size);
        System.out.println("================");

        Polynomial tmp_prev = new Polynomial(solution_trend);
        solution_trend.stepB((solution_trend.getB() - previous_trend.getB()) * step_size * improvement_rate);
        solution_trend.stepC((solution_trend.getC() - previous_trend.getC()) * step_size * improvement_rate);
        previous_trend = tmp_prev;
        previous_error = error;
        displaySolutionTrend();
    }

    @FXML
    public void play_stop(ActionEvent actionEvent) {
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
        solution_display.setData(FXCollections.observableArrayList());
        for (int i = 0; i < dataset_size_slider.getValue(); ++i) {
            solution_display.getData().add( new XYChart.Data(i, solution_trend.solve_for(i)));
        }
        display_graph.getData().set(1,solution_display);
    }
}
