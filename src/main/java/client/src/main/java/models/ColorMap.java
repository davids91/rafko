package models;

import javafx.scene.paint.Color;

public class ColorMap {
    private static double minRange = -1.0;
    private static double maxRange = 1.0;
    public static void range(double minRange, double maxRange){
        minRange = Math.min(minRange,maxRange);
        maxRange = Math.max(minRange,maxRange);
    }
    public static Color getColor(double number){
        double r = number / (maxRange - minRange);
        double g = number / (maxRange - minRange);
        double b = number / (maxRange - minRange);
        if(number < minRange) b = 1.0;
        if(number > maxRange) r = 1.0;
        r = Math.min(Math.max(0.0,r),1.0);
        g = Math.min(Math.max(0.0,g),1.0);
        b = Math.min(Math.max(0.0,b),1.0);
        return Color.color(r,g,b);
    }
}
