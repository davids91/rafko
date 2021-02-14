package com.crystalline.aether.models;

import com.badlogic.gdx.graphics.Color;

public class Materials {
    public enum names{
        Air,
        Fire,
        Null_Crystal,
        Water,
        Earth
    };

    public static final Color[] colors = {
        Color.SKY,
        Color.ORANGE,
        Color.PURPLE,
        Color.BLUE,
        Color.BROWN
    };

    /**!Note: The ratio of the two values define the material states. Reality tries to "stick" to given ratios,
     * The difference in s radiating away.  */
    public static final float [] nether_ratios = {
            0.25f, /* Air */
            0.5f, /* Fire */
            1.0f, /* No */
            2.0f, /* Water */
            4.0f, /* Earth */
    };

}
