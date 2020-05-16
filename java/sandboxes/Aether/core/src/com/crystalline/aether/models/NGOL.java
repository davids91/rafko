package com.crystalline.aether.models;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.Pixmap;

import java.text.Format;
import java.util.Random;


/**
 * NGOL meaning "Not Game Of Life" is an adaptation of the original game of life,
 * using summaries and thresholds analougous to the original games.
 * TODO: Use GPU instead of pixmap
 */
public class NGOL {

    @FunctionalInterface
    public interface NGOL_RULE{
        Color execute(Color pixel, float proxR, float proxG, float proxB, int x, int y);
    }

    NGOL_RULE custom_rule = null;
    Pixmap[] ngolBuffer;
    int usedBuf = 0;

    /**
     * Interesting ranges:
     * - 0.7 ; 1.0 - Square trip
     * - 1.0 ; 1.0 - Trip
     * - 2.0 ; 3.0 - Game Of Life ??
     * - 2.5 ; 4.0 - New Game of life 2 ??
     * - 3.0 ; 5.0 - Building Squares
     * - 3.0 ; 7.0 - New Game of Life??
     * - 4.0 ; 9.0 - Stable land
     */
    float underPopThr = 4f;
    float overPopThr = 9f;

    final Random rnd = new Random();

    public NGOL(int width, int height, float uThr, float oThr, NGOL_RULE custom_rule_) {
        this(width,height,uThr,oThr);
        custom_rule = custom_rule_;
    }

    public NGOL(int width, int height, float uThr, float oThr) {
        ngolBuffer = new Pixmap[2];
        ngolBuffer[0] = new Pixmap(width, height, Pixmap.Format.RGBA8888);
        ngolBuffer[0].setColor(Color.BLACK);
        ngolBuffer[0].fill();
        ngolBuffer[1] = new Pixmap(width, height, Pixmap.Format.RGBA8888);
        ngolBuffer[1].setColor(Color.BLACK);
        ngolBuffer[1].fill();
        underPopThr = uThr;
        overPopThr = oThr;
    }

    Color envPixColor;
    float r = 0; float g = 0; float b = 0;
    float proxR = 0; float proxG = 0; float proxB = 0;

    public void loop(){
        for(int i = 0; i < ngolBuffer[usedBuf].getWidth(); i++){
            for(int j = 0; j < ngolBuffer[usedBuf].getHeight(); j++){
                envPixColor = new Color(ngolBuffer[usedBuf].getPixel(i,j));
                r = envPixColor.r; g = envPixColor.g; b = envPixColor.b;

                proxR = 0; proxG = 0; proxB = 0;

                for(int inItX = Math.max(0,i-1); inItX <= Math.min((ngolBuffer[usedBuf].getWidth()-1),i+1); inItX++){
                    for(int inItY = Math.max(0,j-1); inItY <= Math.min((ngolBuffer[usedBuf].getHeight()-1),j+1); inItY++){
                        envPixColor = new Color(ngolBuffer[usedBuf].getPixel(inItX,inItY));
                        proxR += envPixColor.r;
                        proxG += envPixColor.g;
                        proxB += envPixColor.b;
                    }
                }

                if((underPopThr > proxR)||(overPopThr < proxR))r = 0; /* under + overpopulation */
                else if((underPopThr <= proxR)&&(overPopThr >= proxR))r = proxR / ((underPopThr+overPopThr)/2.0f); /* re-production */
                else r = 0;

                if((underPopThr > proxG)||(overPopThr < proxG))g = 0; /* under + overpopulation */
                else if((underPopThr <= proxG)&&(overPopThr >= proxG))g =  proxG / ((underPopThr+overPopThr)/2.0f); /* re-production */
                else g = 0;

                if((underPopThr > proxB)||(overPopThr < proxB))b = 0; /* under + overpopulation */
                else if((underPopThr <= proxB)&&(overPopThr >= proxB))b = proxB / ((underPopThr+overPopThr)/2.0f); /* re-production */
                else b = 0;

                envPixColor.set( Math.min(1.0f, r),Math.min(1.0f, g),Math.min(1.0f, b),1.0f);
                if(null != custom_rule)envPixColor = custom_rule.execute(envPixColor,proxR,proxG,proxB,i,j);

                ngolBuffer[(usedBuf + 1)%2].drawPixel(i, j, Color.rgba8888(envPixColor));
            }
        }
        usedBuf = (usedBuf + 1)%2;
    }

    public void reset(float addR, float addG, float addB){
        usedBuf = 0;
        for(int i = 0; i < ngolBuffer[usedBuf].getWidth(); i++)
        {
            for(int j = 0; j < ngolBuffer[usedBuf].getHeight(); j++)
            { /* fill up data with random */
                float pixelStrength = rnd.nextFloat() ;
                Color pixColor = new Color(ngolBuffer[usedBuf].getPixel(i,j));
                pixColor.set(
                    pixelStrength * addR,
                    pixelStrength * addG,
                    pixelStrength * addB,
                    pixColor.a
                );
                ngolBuffer[usedBuf].drawPixel(i,j,Color.rgba8888(pixColor));
            }
        }
    }

    public void setThresholds(float uThr, float oThr){
        underPopThr = uThr;
        overPopThr = oThr;
    }

    public void setRule(NGOL_RULE new_rule){
        custom_rule = new_rule;
    }
    public void setUnderPopThr(float uThr){
        underPopThr = uThr;
    }
    public void setOverPopThr(float oThr){
        overPopThr = oThr;
    }
    public Pixmap getBoard(){
        return ngolBuffer[usedBuf];
    }

}
