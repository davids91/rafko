package Models;

import java.util.ArrayList;

public class ErrorFunction {
    public static double getErrorValue(ArrayList<Double> dataset, Polynomial function){
        double error = 0;
        for (int i = 0; i < dataset.size(); ++i) {
            error += Math.pow(dataset.get(i) - function.solve_for(i),2); /* Squared loss */
        }
        return error;
    }

}
