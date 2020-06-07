package com.aether.ngol.models;

import com.aether.ngol.services.MouseInputProcessor;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.InputMultiplexer;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.TextureAtlas;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.EventListener;
import com.badlogic.gdx.scenes.scene2d.Stage;
import com.badlogic.gdx.scenes.scene2d.ui.*;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;
import com.badlogic.gdx.utils.viewport.ExtendViewport;

import java.util.HashMap;

public class MainLayout {
    private Stage stage;
    private Table main_layout;
    private TextButton reset_button;
    private TextButton step_button;
    private Slider uThrSlider;
    private Slider oThrSlider;
    private Slider speed_slider;
    private Label my_label;
    private Label my_label2;
    private Label uThr_label;
    private Label oThr_label;
    private Label speed_label;
    private Minimap minimap;
    private BrushPanel brush_panel;
    private Image capture_mouse;
    private Image touch_me;

    public float getSpeed(){
        return speed_slider.getValue();
    }
    public Minimap getMinimap(){return  minimap;}
    public BrushPanel getBrushPanel(){return brush_panel;}
    public Stage getStage(){
        return stage;
    }

    public MainLayout(HashMap<String, EventListener> actions, HashMap<String, Float> data){
        TextureAtlas ui_atlas = new TextureAtlas("neutralizer-ui.atlas");
        TextureAtlas extra_atlas = new TextureAtlas("ngol_ui.atlas");
        BitmapFont bitmapFont = new BitmapFont(Gdx.files.internal("font-export.fnt"), ui_atlas.findRegion("font-export"));
        Skin used_skin = new Skin();
        used_skin.addRegions(ui_atlas);
        used_skin.addRegions(extra_atlas);

        stage = new Stage(new ExtendViewport(Gdx.graphics.getWidth(),Gdx.graphics.getHeight()));
        touch_me = new Image();
        touch_me.setFillParent(true);
        touch_me.addListener(actions.get("touch"));
        stage.addActor(touch_me);

        capture_mouse = new Image(used_skin.getDrawable("capture_icon"));
        capture_mouse.setSize(64,64);
        capture_mouse.setVisible(false);
        stage.addActor(capture_mouse);

        main_layout = new Table();
        main_layout.setFillParent(true);
        main_layout.setFillParent(true);
        main_layout.top().left();

        Table control_panel = new Table();

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

        reset_button = new TextButton("Reset", textButtonStyle);
        if(null != actions.get("reset"))
            reset_button.addListener(actions.get("reset"));

        step_button = new TextButton("Step", textButtonStyle);
        if(null != actions.get("step"))
            step_button.addListener(actions.get("step"));

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

        minimap = new Minimap(
            used_skin, new Vector2(data.get("ngol-width"),data.get("ngol-height"))
        );
        brush_panel = new BrushPanel(used_skin, actions);

        control_panel.add(my_label);
        control_panel.add(uThrSlider);
        control_panel.add(uThr_label);
        control_panel.add(oThrSlider);
        control_panel.add(oThr_label);
        control_panel.row().fill();
        control_panel.add(my_label2);
        control_panel.add(speed_slider);
        control_panel.add(speed_label);
        control_panel.add(reset_button).expand();
        control_panel.add(step_button).expand();
        control_panel.row().fill();
        control_panel.add(brush_panel).left().padLeft(-25);

        stage.addActor(main_layout);
        main_layout.add(control_panel).top().left().expandX();
        main_layout.add(minimap).prefSize(128,128).top().left();
        minimap.layout();
    }

    public void setInputProcessor(){
        MouseInputProcessor mySP = new MouseInputProcessor(new MouseInputProcessor.My_scroll_action_interface() {
            @Override
            public boolean scrollAction(int scrollValue) {
                getMinimap().adjust_zoom(-0.2f * scrollValue);
                return false;
            }

            @Override
            public boolean mouseMoveAction(int screenX, int screenY) {
                if(!Gdx.input.isKeyPressed(Input.Keys.TAB))
                capture_mouse.setPosition(
                screenX - capture_mouse.getWidth()/2,
                screenY - capture_mouse.getHeight()/2
                );
                return false;
            }

            @Override
            public boolean touchDownAction(int screenX, int screenY, int pointer, int button) {
                capture_mouse.setVisible(false);
                return false;
            }
        });
        InputMultiplexer myInput = new InputMultiplexer();
        myInput.addProcessor(mySP);
        myInput.addProcessor(main_layout.getStage());
        Gdx.input.setInputProcessor(myInput);
    }

    public void layout(){
        stage.getViewport().update(Gdx.graphics.getWidth(), Gdx.graphics.getHeight(), true);
        minimap.layout();
    }

    public void startCapture(){
        capture_mouse.setVisible(true);
    }
}
