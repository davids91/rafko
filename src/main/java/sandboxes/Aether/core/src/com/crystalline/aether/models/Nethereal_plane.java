package com.crystalline.aether.models;

import com.badlogic.gdx.graphics.Color;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Nethereal_plane extends Reality_plane{

    private float [][] aether_values; /* Stationary substance */
    private float [][] radiated_aether; /* temporary variable */
    private float [][] nether_values; /* Moving substance */
    private float [][] radiated_nether; /* temporary variable */

    private final float tendency_to_stabilize = 0.7f;

    private final Random rnd = new Random();

    public Nethereal_plane(int sizeX_, int sizeY_){
        super(sizeX_,sizeY_);
        aether_values = new float[sizeX][sizeY];
        radiated_aether = new float[sizeX][sizeY];
        nether_values = new float[sizeX][sizeY];
        radiated_nether = new float[sizeX][sizeY];
        randomize();
    }

    public void randomize(){
        for(int x = 0;x < sizeX; ++x){
            for(int y = 0; y < sizeY; ++y){
                aether_values[x][y] = rnd.nextFloat() * 30.0f * Materials.nether_ratios[Materials.nether_ratios.length-1];
                nether_values[x][y] = rnd.nextFloat() * 30.0f * 1.0f;
            }
        }
    }

    public void attach_to(Elemental_plane plane){
        for(int x = 0;x < sizeX; ++x){
            for(int y = 0; y < sizeY; ++y){
                nether_values[x][y] = rnd.nextFloat() * 30.0f + 0.01f;
                aether_values[x][y] = Materials.nether_ratios[plane.element_at(x,y).ordinal()] * nether_values[x][y];
            }
        }
    }

    public void main_loop(float step_value){
        float ratio;
        float ratio_distance = 1.0f;
        float target_value;
        float delta;
        int converge_to = Materials.nether_ratios.length;
//        System.out.println("Ratios:");
//        for(int x = 0;x < sizeX; ++x){
//            for(int y = 0; y < sizeY; ++y) {
//                System.out.print("["
//                        + aether_values[x][y] + "/" +nether_values[x][y] +" = "
//                        + aether_values[x][y] / nether_values[x][y]
//                    + "]");
//            }
//            System.out.print("\n");
//        }
        for(int x = 0;x < sizeX; ++x){
            for(int y = 0; y < sizeY; ++y){
                /* Calculate ratio of substances */
                ratio = aether_values[x][y] / nether_values[x][y];
                ratio_distance = Float.MAX_VALUE;
                converge_to = Materials.nether_ratios.length;

                /* decide the nearest "ratio" to stick to  */
                for(int i = 0; i < Materials.nether_ratios.length; ++i){
                    if(Math.abs(ratio - Materials.nether_ratios[i]) < ratio_distance){
                        ratio_distance = Math.abs(ratio - Materials.nether_ratios[i]);
                        converge_to = i;
                    }
                }

                /* if It's sticking to a ratio */
                if(
                    (Materials.nether_ratios.length > converge_to)
                    &&(Materials.names.Null_Crystal.ordinal() != converge_to)
                ){
                    /* Radiate out some materials to be closer to the ratio */
                    if(0.01f < Math.abs(ratio - Materials.nether_ratios[converge_to]))
                    if(aether_values[x][y] >= nether_values[x][y]){
                        /* aether value should be nether value multiplied by ratio */
                        target_value = nether_values[x][y] * Materials.nether_ratios[converge_to];
                        delta = target_value - aether_values[x][y];
                        if(0 > delta)aether_values[x][y] += step_value * delta;
                        radiated_aether[x][y] = step_value * delta;

                        /* nether value should be aether value multiplied by 1/ratio */
                        target_value = aether_values[x][y] / Materials.nether_ratios[converge_to];
                        delta = target_value - nether_values[x][y];
                        if(0 > delta)nether_values[x][y] += step_value  * delta * tendency_to_stabilize;
                        radiated_nether[x][y] = step_value * delta * tendency_to_stabilize;
                    }else{
                        /* aether value should be nether value multiplied by 1/ratio */
                        target_value = nether_values[x][y] / Materials.nether_ratios[converge_to];
                        delta = target_value - aether_values[x][y];
                        if(0 > delta)aether_values[x][y] += step_value * delta;
                        radiated_aether[x][y] = step_value * delta;

                        /* nether value should be aether value multiplied by ratio */
                        target_value = aether_values[x][y] * Materials.nether_ratios[converge_to];
                        delta = target_value - nether_values[x][y];
                        if(0 > delta)nether_values[x][y] += step_value  * delta * tendency_to_stabilize;
                        radiated_nether[x][y] = step_value * delta * tendency_to_stabilize;
                    }
                    if(0 > aether_values[x][y]) aether_values[x][y] = 0;
                    if(0.1 >= nether_values[x][y]) nether_values[x][y] = 0.001f;
                }

                if(ratio < Materials.nether_ratios[Materials.names.Air.ordinal()] * tendency_to_stabilize){ /* If it's volatile */
                    delta = nether_values[x][y] - aether_values[x][y];
                    //delta = (float)Math.pow(delta,1/aether_values[x][y]);
                    for(int inItX = Math.max(0,x-1); inItX <= Math.min((sizeX-1),x+1); inItX++){
                        for(int inItY = Math.max(0,y-1); inItY <= Math.min((sizeY-1),y+1); inItY++){
                            aether_values[inItX][inItY] += aether_values[x][y]*delta * 0.001f * tendency_to_stabilize;
                            nether_values[inItX][inItY] += nether_values[x][y]*delta * 0.001f * tendency_to_stabilize;
                            aether_values[x][y] -= aether_values[x][y]*delta * 0.001f;
                            nether_values[x][y] -= nether_values[x][y]*delta * 0.001f;
                        }
                    }
                }
            }
        }

        for(int x = 0;x < sizeX; ++x) {
            for (int y = 0; y < sizeY; ++y) {
                /* Add some values from the neighbours */
                for (int nx = Math.max(0, (x - 1)); nx < Math.min(sizeX, x + 1); ++nx) {
                    for (int ny = Math.max(0, (y - 1)); ny < Math.min(sizeX, y + 1); ++ny) {
                        aether_values[x][y] += radiated_aether[nx][ny] * 0.001f;
                        nether_values[x][y] += radiated_nether[nx][ny] * 0.001f;
                    }
                }
            }
        }
    }

    public float ratio_at(int posX, int posY){
        return aether_values[posX][posY]/nether_values[posX][posY];
    }
    public float aether_value_at(int posX, int posY){
        return aether_values[posX][posY];
    }
    public float nether_value_at(int posX, int posY){
        return nether_values[posX][posY];
    }
    public void add_nether_to(int posX, int posY, float value){
        nether_values[posX][posY] += value;
    }

}
