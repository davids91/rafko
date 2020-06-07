package com.aether.ngol.models;

import com.aether.ngol.services.MouseInputProcessor;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.InputMultiplexer;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.TextureAtlas;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.math.Vector3;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.EventListener;
import com.badlogic.gdx.scenes.scene2d.Stage;
import com.badlogic.gdx.scenes.scene2d.ui.*;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;
import com.badlogic.gdx.utils.viewport.ExtendViewport;

import java.util.HashMap;

public class MainLayout {
    Stage stage;
    Table main_layout;
    TextButton reset_button;
    TextButton step_button;
    TextButton play_button;
    Slider uThrSlider;
    Slider oThrSlider;
    Slider speed_slider;
    Label my_label;
    Label my_label2;
    Label uThr_label;
    Label oThr_label;
    Label speed_label;
    Minimap minimap;
    BrushPanel brush_panel;
    Image capture_mouse;
    Image touch_me;

    boolean capturing = false;

    public float getSpeed(){
        return speed_slider.getValue();
    }
    public Minimap getMinimap(){return  minimap;}
    public BrushPanel getBrushPanel(){return brush_panel;}
    public Stage getStage(){
        return stage;
    }

    public MainLayout(final HashMap<String, EventListener> actions, HashMap<String, Float> data){
        TextureAtlas ui_atlas = new TextureAtlas("neutralizer-ui.atlas");
        TextureAtlas extra_atlas = new TextureAtlas("ngol_ui.atlas");
        BitmapFont bitmapFont = new BitmapFont(Gdx.files.internal("font-export.fnt"), ui_atlas.findRegion("font-export"));
        Skin used_skin = new Skin();
        used_skin.addRegions(ui_atlas);
        used_skin.addRegions(extra_atlas);

        stage = new Stage(new ExtendViewport(Gdx.graphics.getWidth(),Gdx.graphics.getHeight()));

        capture_mouse = new Image(used_skin.getDrawable("capture_icon"));
        capture_mouse.setSize(64,64);
        capture_mouse.setVisible(false);
        stage.addActor(capture_mouse);

        touch_me = new Image();
        touch_me.setFillParent(true);
        touch_me.addListener(actions.get("touch"));
        stage.addActor(touch_me);

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

        play_button = new TextButton("Play", textButtonStyle);
        play_button.addListener(new ChangeListener() {
            @Override
            public void changed(ChangeEvent event, Actor actor) {
                if(0 == speed_slider.getValue()){
                    play_button.setText("Pause");
                    speed_slider.setValue(0.05f);
                }else{
                    play_button.setText("Play");
                    speed_slider.setValue(0.0f);
                }
            }
        });

        my_label = new Label("Life ranges:" , label_style);
        uThr_label = new Label("1.9" , label_style);
        oThr_label = new Label("2.9" , label_style);
        my_label2 = new Label("Speed:" , label_style);
        speed_label = new Label("Stop." , label_style);

        uThrSlider = new Slider(0,10,0.1f,false, slider_style);
        uThrSlider.setSize(256,64);
        uThrSlider.setValue(1.9f);
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
        oThrSlider.setValue(2.9f);
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
                else speed_label.setText("Stop.");
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
        control_panel.add(play_button).prefWidth(64);
        control_panel.add(reset_button).prefWidth(64);
        control_panel.add(step_button).prefWidth(64);
        control_panel.row().fill();
        control_panel.add(brush_panel).left().padLeft(-25);

        stage.addActor(main_layout);
        main_layout.add(control_panel).top().left().expandX();
        main_layout.add(minimap).size(128,128).top().left();
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
                if(!Gdx.input.isKeyPressed(Input.Keys.TAB)){
                    Vector3 target_position = getStage().getCamera().unproject(
                        new Vector3(screenX, Gdx.graphics.getHeight() - screenY,0)
                    );
                    capture_mouse.setPosition(
                    target_position.x - capture_mouse.getWidth()/2,
                    target_position.y - capture_mouse.getHeight()/2
                    );
                }
                return false;
            }

            @Override
            public boolean touchDownAction(int screenX, int screenY, int pointer, int button) {
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

    public Texture get_brush(){
        if(
            (null != getBrushPanel().get_selected_brush())
            &&(!capturing)
        ){
            return new Texture(getBrushPanel().get_selected_brush());
        }else return null;
    }

    public boolean is_capturing(){
        return capturing;
    }

    public void capture(Texture tex){
        if(capturing){
            brush_panel.update_selected_brush(tex);
            stop_capture();
        }
    }

    public void start_capture(){
        if(!capturing){
            capture_mouse.setVisible(true);
            brush_panel.start_capture();
            capturing = true;
        }
    }

    public void stop_capture(){
        if(capturing){
            capture_mouse.setVisible(false);
            brush_panel.stop_capture();
            capturing = false;
        }
    }

}
