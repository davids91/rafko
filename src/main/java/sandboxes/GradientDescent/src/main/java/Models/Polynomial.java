package Models;

import java.util.Random;

/**
 * AX2 + Bx + C = ?
 */
public class Polynomial {

    public Polynomial(Polynomial other){
        varA = other.varA;
        varB = other.varB;
        varC = other.varC;
    }
    public Polynomial(Polynomial other, double scale){
        varA = other.varA * scale;
        varB = other.varB * scale;
        varC = other.varC * scale;
    }

    public Polynomial(Random rnd, double scale){
        varA = (0.5 - rnd.nextDouble()) * scale;
        varB = (0.5 - rnd.nextDouble()) * scale;
        varC = (0.5 - rnd.nextDouble()) * scale * 10;
    }

    public double solve_for(double x){
        return varA * x*x + varB * x + varC;
    }

    private double varA = 0;

    public double getA() {
        return varA;
    }

    public double getB() {
        return varB;
    }

    public double getC() {
        return varC;
    }

    private double varB = 0;
    private double varC = 0;

    public void setA(double value){
        varA = value;
    }

    public void setB(double value){
        varB = value;
    }

    public void setC(double value){
        varC = value;
    }

    public void stepA(double step) {
        this.varA += step;
    }
    public void stepB(double step) {
        this.varB += step;
    }
    public void stepC(double step) {
        this.varC += step;
    }
}
