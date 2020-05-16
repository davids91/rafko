package com.crystalline.aether.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Reality {

    private final int sizeX;
    private final int sizeY;
    private float [][] aether_values; /* Stationary substance */
    private float [][] radiated_aether; /* temporary variable */
    private float [][] nether_values; /* Moving substance */
    private float [][] radiated_nether; /* temporary variable */

    private final float tendency_to_stabilize = 0.7f;
    private final float [] material_ratios = {
        2.0f, /* Air */
        4.0f, /* Fire */
        //0.5f, /* No */
        6.0f, /* Water */
        8.0f, /* Earth */
    };

    private final Random rnd = new Random();

    /**!Note: The ratio of the two values define the materials state. Reality tries to "stick" to given ratios,
     * The difference in s radiating away.  */

    public  Reality(int sizeX_, int sizeY_){
        sizeX = sizeX_;
        sizeY = sizeY_;
        aether_values = new float[sizeX][sizeY];
        radiated_aether = new float[sizeX][sizeY];
        nether_values = new float[sizeX][sizeY];
        radiated_nether = new float[sizeX][sizeY];
        randomize();
    }

    public void randomize(){
        for(int x = 0;x < sizeX; ++x){
            for(int y = 0; y < sizeY; ++y){
                aether_values[x][y] = rnd.nextFloat() * 100.0f * material_ratios[material_ratios.length-1];
                nether_values[x][y] = rnd.nextFloat() * 100.0f * 1.0f;
            }
        }
    }

    public void main_loop(float step_value){
        float ratio;
        float ratio_distance = 1.0f;
        float target_value;
        float delta;
        int diverge_to = material_ratios.length;
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

                /* decide the nearest "ratio" to stick to  */
                for(int i = 0; i < material_ratios.length; ++i){
                    if(Math.abs(ratio - material_ratios[i]) < ratio_distance){
                        ratio_distance = Math.abs(ratio - material_ratios[i]);
                        diverge_to = i;
                    }
                }

                /* if It's sticking to a ratio */
                if(material_ratios.length > diverge_to){
                    /* Radiate out some materials to be closer to the ratio */

                    if(aether_values[x][y] > nether_values[x][y]){
                        /* aether value should be nether value multiplied by ratio */
                        target_value = nether_values[x][y] * material_ratios[diverge_to];
                        delta = target_value - aether_values[x][y];
                        aether_values[x][y] += step_value * delta;
                        radiated_aether[x][y] = step_value * delta;

                        /* nether value should be aether value multiplied by 1/ratio */
                        target_value = aether_values[x][y] / material_ratios[diverge_to];
                        delta = target_value - nether_values[x][y];
                        nether_values[x][y] += step_value  * delta * tendency_to_stabilize;
                        radiated_nether[x][y] = step_value * delta * tendency_to_stabilize;
                    }else{
                        /* aether value should be nether value multiplied by 1/ratio */
                        target_value = nether_values[x][y] / material_ratios[diverge_to];
                        delta = target_value - aether_values[x][y];
                        aether_values[x][y] += step_value * delta;
                        radiated_aether[x][y] = step_value * delta;

                        /* nether value should be aether value multiplied by ratio */
                        target_value = aether_values[x][y] * material_ratios[diverge_to];
                        delta = target_value - nether_values[x][y];
                        nether_values[x][y] += step_value  * delta * tendency_to_stabilize;
                        radiated_nether[x][y] = step_value * delta * tendency_to_stabilize;
                    }
                    if(0 > aether_values[x][y]) aether_values[x][y] = 0;
                    if(0.1 >= nether_values[x][y]) nether_values[x][y] = 0.001f;
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

}
