package com.aether.ngol;

import com.aether.ngol.models.MainLayout;
import com.aether.ngol.services.NGOL;
import com.aether.ngol.services.ScrollProcessor;
import com.badlogic.gdx.*;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.Stage;
import com.badlogic.gdx.scenes.scene2d.ui.*;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;
import sun.applet.Main;

import java.util.HashMap;

public class NGOL_Main extends ApplicationAdapter {
	private Stage stage;
	private MainLayout main_layout;

	NGOL ngol;
	float time_delayed = 0;

	@Override
	public void create () {
		HashMap<String,ChangeListener> actions = new HashMap<>();
		actions.put("goBtn",new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				ngol.randomize();
			}
		});
		actions.put("uThrSlider",new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				ngol.setUnderPopThr(((Slider)actor).getValue());
			}
		});
		actions.put("oThrSlider",new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				ngol.setOverPopThr(((Slider)actor).getValue());
			}
		});
		main_layout = new MainLayout(actions);
		stage = main_layout.getUI();

		ScrollProcessor mySP = new ScrollProcessor(new ScrollProcessor.My_scroll_action_interface() {
			@Override
			public void scrollAction(int scrollValue) {
			main_layout.getMinimap().adjust_zoom(-0.1f * scrollValue);
			}
		});
		InputMultiplexer myInput = new InputMultiplexer();
		myInput.addProcessor(mySP);
		myInput.addProcessor(stage);
		Gdx.input.setInputProcessor(myInput);
		Gdx.gl.glClearColor(0.2f, 0.5f, 0.1f, 1);
		ngol = new NGOL(2048,2048, 2f, 3f);
		ngol.randomize();
		main_layout.getMinimap().set_map_image(ngol.getBoard());
	}

	@Override
	public void resize(int width, int height){
		stage.getViewport().update(width, height, true);
	}

	@Override
	public void render () {
		Gdx.gl.glClearColor(0.0f, 0.0f, 0.0f, 1);
		if(!Gdx.input.isKeyPressed(Input.Keys.TAB)){
			if(main_layout.getSpeed() > 0)
			if(time_delayed < main_layout.getSpeed()){
				time_delayed += Gdx.graphics.getDeltaTime();
			}else{
				ngol.loop();
				main_layout.getMinimap().set_map_image(ngol.getBoard());
				time_delayed = 0;
			}
		}

		if(Gdx.input.isKeyPressed(Input.Keys.SPACE)&&Gdx.input.isTouched()){
			ngol.addBrush(
				new Texture(Gdx.files.internal("peener.png")),
					main_layout.getMinimap().get_world_coordinates(new Vector2(Gdx.input.getX(),Gdx.graphics.getHeight() - Gdx.input.getY())),
				new Vector2(64 / main_layout.getMinimap().get_zoom(),64 / main_layout.getMinimap().get_zoom())
			);
		}

		if(Gdx.input.isKeyPressed(Input.Keys.MINUS)){
			main_layout.getMinimap().adjust_zoom(-0.05f);
		}
		if(Gdx.input.isKeyPressed(Input.Keys.PLUS)){
			main_layout.getMinimap().adjust_zoom(+0.05f);
		}

		Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
		stage.act(Gdx.graphics.getDeltaTime());

		stage.getBatch().begin();
		stage.getBatch().draw(
			ngol.getBoard(),
				-main_layout.getMinimap().get_position(
				).x,
				-main_layout.getMinimap().get_position(
				).y,
			Gdx.graphics.getWidth()*main_layout.getMinimap().get_zoom(),
			Gdx.graphics.getHeight()*main_layout.getMinimap().get_zoom()
		);
		stage.getBatch().end();

		stage.draw();
	}
	
	@Override
	public void dispose () {
		stage.dispose();
	}
}
