package com.aether.ngol.models;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.Sprite;
import com.badlogic.gdx.graphics.g2d.TextureAtlas;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.Stage;
import com.badlogic.gdx.scenes.scene2d.ui.*;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;
import com.badlogic.gdx.scenes.scene2d.utils.Drawable;

import java.util.HashMap;

public class MainLayout {
    private static Stage stage;
    private static Table main_layout;
    private static TextButton goBtn;
    private static Slider uThrSlider;
    private static Slider oThrSlider;
    private static Slider speed_slider;
    private static Label my_label;
    private static Label my_label2;
    private static Label uThr_label;
    private static Label oThr_label;
    private static Label speed_label;
    private static Image minimap;
    private static Image minimap_border;

    public static void set_minimap_image(Drawable minimap_image){
        minimap.setDrawable(minimap_image);
    }

    public static float getSpeed(){
        return speed_slider.getValue();
    }

    public static Stage getUI(HashMap<String, ChangeListener> actions){
        TextureAtlas my_atlas = new TextureAtlas("neutralizer-ui.atlas");
        BitmapFont bitmapFont = new BitmapFont(Gdx.files.internal("font-export.fnt"), my_atlas.findRegion("font-export"));

        stage = new Stage();
        Skin used_skin = new Skin();
        used_skin.addRegions(my_atlas);

        TextButton.TextButtonStyle textButtonStyle = new TextButton.TextButtonStyle();
        textButtonStyle.font = bitmapFont;
        textButtonStyle.up = used_skin.getDrawable("button");
        textButtonStyle.down = used_skin.getDrawable("button-pressed");

        Slider.SliderStyle slider_style = new Slider.SliderStyle();
        slider_style.knob = used_skin.getDrawable("slider-knob-horizontal");
        slider_style.knobDown = used_skin.getDrawable("slider-knob-pressed-horizontal");
        slider_style.knobOver = used_skin.getDrawable("slider-knob-horizontal");
        slider_style.background = used_skin.getDrawable("scrollbar-horizontal");

        Label.LabelStyle label_style = new Label.LabelStyle();
        label_style.font = bitmapFont;
        label_style.fontColor = Color.RED;

        goBtn = new TextButton("Reset", textButtonStyle);
        goBtn.setSize(128,128);

        if(null != actions.get("goBtn"))
            goBtn.addListener(actions.get("goBtn"));

        my_label = new Label("Life ranges:" , label_style);
        uThr_label = new Label("2" , label_style);
        oThr_label = new Label("3" , label_style);
        my_label2 = new Label("Speed:" , label_style);
        speed_label = new Label("Stopped" , label_style);

        uThrSlider = new Slider(0,10,0.1f,false, slider_style);
        uThrSlider.setSize(256,64);
        uThrSlider.setValue(2);
        if(null != actions.get("uThrSlider"))
            uThrSlider.addListener(actions.get("uThrSlider"));
        uThrSlider.addListener(new ChangeListener() {
            @Override
            public void changed(ChangeEvent event, Actor actor) {
                uThr_label.setText(String.format("%.2f",uThrSlider.getValue()));
            }
        });

        oThrSlider = new Slider(0,10,0.1f,false, slider_style);
        oThrSlider.setSize(256,64);
        oThrSlider.setValue(3);
        if(null != actions.get("oThrSlider"))
            oThrSlider.addListener(actions.get("oThrSlider"));
        oThrSlider.addListener(new ChangeListener() {
            @Override
            public void changed(ChangeEvent event, Actor actor) {
                oThr_label.setText(String.format("%.2f",oThrSlider.getValue()));
            }
        });

        speed_slider = new Slider(0,2,0.05f,false, slider_style);
        speed_slider.setSize(256,64);
        speed_slider.setValue(0.0f);
        if(null != actions.get("speed_slider"))
            speed_slider.addListener(actions.get("speed_slider"));
        speed_slider.addListener(new ChangeListener() {
            @Override
            public void changed(ChangeEvent event, Actor actor) {
                if(speed_slider.getValue() > 0)
                    speed_label.setText(Float.toString(1/speed_slider.getValue()));
                else speed_label.setText("Stopped");
            }
        });

        minimap_border = new Image(used_skin.getRegion("minimap"));
        minimap_border.setFillParent(true);
        minimap = new Image();
        minimap.setFillParent(true);
        Stack minimapStack = new Stack();
        minimapStack.add(minimap);
        minimapStack.add(minimap_border);

        main_layout = new Table();
        Table control_panel = new Table();
        control_panel.add(my_label);
        control_panel.add(uThrSlider);
        control_panel.add(uThr_label);
        control_panel.add(oThrSlider);
        control_panel.add(oThr_label);
        control_panel.row();
        control_panel.add(my_label2);
        control_panel.add(speed_slider);
        control_panel.add(speed_label);
        control_panel.add(goBtn);

        main_layout.setFillParent(true);
        main_layout.setDebug(false);
        main_layout.top().left();

        stage.addActor(main_layout);
        main_layout.add(control_panel).top().left().expandX();
        main_layout.add(minimapStack).prefSize(64,64);
        return stage;
    }
}
