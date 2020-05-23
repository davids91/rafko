package com.aether.ngol;

import com.aether.ngol.services.NGOL;
import com.badlogic.gdx.ApplicationAdapter;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.TextureAtlas;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.graphics.glutils.ShaderProgram;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.Stage;
import com.badlogic.gdx.scenes.scene2d.ui.*;
import com.badlogic.gdx.scenes.scene2d.ui.Label;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;

public class NGOL_Main extends ApplicationAdapter {
	private Stage stage;

	private VerticalGroup main_layout;
	private TextButton goBtn;
	private Slider uThrSlider;
	private Slider oThrSlider;
	private Slider speed_slider;
	private Skin used_skin;
	private Label my_label;
	private Label my_label2;
	private Label uThr_label;
	private Label oThr_label;
	private Label speed_label;

	NGOL ngol;
	float time_delayed = 0;

	@Override
	public void create () {
		TextureAtlas my_atlas = new TextureAtlas("neutralizer-ui.atlas");
		BitmapFont bitmapFont = new BitmapFont(Gdx.files.internal("font-export.fnt"), my_atlas.findRegion("font-export"));

		stage = new Stage();
		used_skin = new Skin();
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
		goBtn.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				ngol.randomize();
			}
		});

		my_label = new Label("Population Thresholds:" , label_style);
		uThr_label = new Label("4" , label_style);
		oThr_label = new Label("9" , label_style);
		my_label2 = new Label("Speed:" , label_style);
		speed_label = new Label("Stopped" , label_style);

		uThrSlider = new Slider(0,10,0.1f,false, slider_style);
		uThrSlider.setSize(256,64);
		uThrSlider.setValue(4);
		uThrSlider.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				uThr_label.setText(Float.toString(uThrSlider.getValue()));
				ngol.setUnderPopThr(uThrSlider.getValue());
			}
		});

		oThrSlider = new Slider(0,10,0.1f,false, slider_style);
		oThrSlider.setSize(256,64);
		oThrSlider.setValue(9);
		oThrSlider.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				oThr_label.setText(Float.toString(oThrSlider.getValue()));
				ngol.setOverPopThr(oThrSlider.getValue());
			}
		});

		speed_slider = new Slider(0,2,0.05f,false, slider_style);
		speed_slider.setSize(256,64);
		speed_slider.setValue(0.0f);
		speed_slider.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				if(speed_slider.getValue() > 0)
				speed_label.setText(Float.toString(1/speed_slider.getValue()));
				else speed_label.setText("Stopped");
			}
		});

		main_layout = new VerticalGroup();
		main_layout.setFillParent(true);
		main_layout.setDebug(true);
		main_layout.top().left();

		stage.addActor(main_layout);
		main_layout.addActor(goBtn);
		main_layout.addActor(my_label);
		main_layout.addActor(uThrSlider);
		main_layout.addActor(uThr_label);
		main_layout.addActor(oThrSlider);
		main_layout.addActor(oThr_label);
		main_layout.addActor(my_label2);
		main_layout.addActor(speed_slider);
		main_layout.addActor(speed_label);

		Gdx.input.setInputProcessor(stage);
		Gdx.gl.glClearColor(0.2f, 0.5f, 0.1f, 1);
		ngol = new NGOL(64,64, uThrSlider.getValue(), oThrSlider.getValue());
		ngol.randomize();
	}

	@Override
	public void resize(int width, int height){
		stage.getViewport().update(width, height, true);
	}

	@Override
	public void render () {
		/* Remove this--> */Gdx.gl.glClearColor(0.0f, 0.0f, 0.0f, 1);
		if(!Gdx.input.isKeyPressed(Input.Keys.TAB)){
			if(speed_slider.getValue() > 0)
			if(time_delayed < speed_slider.getValue()){
				time_delayed += Gdx.graphics.getDeltaTime();
			}else{
				ngol.loop();
				time_delayed = 0;
			}
		}
		Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
		stage.act(Gdx.graphics.getDeltaTime());

		stage.getBatch().begin();
		stage.getBatch().draw(ngol.getBoard(),0,0,Gdx.graphics.getWidth(),Gdx.graphics.getHeight());
		stage.getBatch().end();

		stage.draw();
	}
	
	@Override
	public void dispose () {
		stage.dispose();
	}
}
