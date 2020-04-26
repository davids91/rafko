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
        displaySolutionTrend();
    }

    private double gradientA = 0.0;
    private double gradientB = 0.0;
    private double gradientC = 0.0;

    private double distanceA = 10;
    private double distanceB = 10;
    private double distanceC = 10;

    private double error = 1.0;

    @FXML
    public void step() { /* this is where the magic will happen */
        /* Calculate the gradient using every sample in the dataset */

        /* Take a weight */
        /* calculate the error with (w+h) */
        Polynomial trend_modA = new Polynomial(solution_trend);
        Polynomial trend_modB = new Polynomial(solution_trend);
        Polynomial trend_modC = new Polynomial(solution_trend);
        trend_modA.stepA(learning_rate_slider.getValue());
        trend_modB.stepB(learning_rate_slider.getValue());
        trend_modC.stepC(learning_rate_slider.getValue());
//        trend_modA.stepA(distanceA);
//        trend_modB.stepB(distanceB);
//        trend_modC.stepC(distanceC);

                gradientA = (
                        (ErrorFunction.getErrorValue(dataset, trend_modA)
                                - ErrorFunction.getErrorValue(dataset, solution_trend))
                );//)/learning_rate_slider.getValue();

                gradientB = (
                        (ErrorFunction.getErrorValue(dataset, trend_modB)
                                - ErrorFunction.getErrorValue(dataset, solution_trend))
                );//)/learning_rate_slider.getValue();

                gradientC = (
                        (ErrorFunction.getErrorValue(dataset, trend_modC)
                                - ErrorFunction.getErrorValue(dataset, solution_trend))
                );//)/learning_rate_slider.getValue();
        /* Calculate the gradient in respect to w */
        error = ErrorFunction.getErrorValue(dataset,solution_trend);
        solution_trend.stepA(-gradientA * learning_rate_slider.getValue());
        solution_trend.stepB(-gradientB * learning_rate_slider.getValue());
        solution_trend.stepC(-gradientC * learning_rate_slider.getValue());
        System.out.println("Gradients: A:" + gradientA + "; B: "+ gradientB +"; C:" + gradientC + ";");
        //System.out.println( "new C: " + solution_trend.getC());
        System.out.println("deviation: " + error);

        displaySolutionTrend();
        if(
            (
                (Math.abs(error) < learning_rate_slider.getValue())
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
