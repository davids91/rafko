package com.aether.ngol;

import com.aether.ngol.models.MainLayout;
import com.aether.ngol.services.NGOL;
import com.aether.ngol.services.ScrollProcessor;
import com.badlogic.gdx.*;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.Stage;
import com.badlogic.gdx.scenes.scene2d.ui.*;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;
import com.badlogic.gdx.scenes.scene2d.utils.TextureRegionDrawable;

import java.util.HashMap;

public class NGOL_Main extends ApplicationAdapter {
	private Stage stage;

	NGOL ngol;
	float time_delayed = 0;
	float zoom_value = 1.0f;

	@Override
	public void create () {
		HashMap<String,ChangeListener> actions = new HashMap<String, ChangeListener>();
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
		stage = MainLayout.getUI(actions);

		ScrollProcessor mySP = new ScrollProcessor(new ScrollProcessor.My_scroll_action_interface() {
			@Override
			public void scrollAction(int scrollValue) {
				zoom_value -= 1.0f * scrollValue;
			}
		});
		InputMultiplexer myInput = new InputMultiplexer();
		myInput.addProcessor(mySP);
		myInput.addProcessor(stage);
		Gdx.input.setInputProcessor(myInput);
		Gdx.gl.glClearColor(0.2f, 0.5f, 0.1f, 1);
		ngol = new NGOL(2048,2048, 2f, 3f);
		ngol.randomize();
		MainLayout.set_minimap_image(new TextureRegionDrawable(ngol.getBoard()));
	}

	@Override
	public void resize(int width, int height){
		stage.getViewport().update(width, height, true);
	}

	@Override
	public void render () {
		Gdx.gl.glClearColor(0.0f, 0.0f, 0.0f, 1);
		if(!Gdx.input.isKeyPressed(Input.Keys.TAB)){
			if(MainLayout.getSpeed() > 0)
			if(time_delayed < MainLayout.getSpeed()){
				time_delayed += Gdx.graphics.getDeltaTime();
			}else{
				ngol.loop();
				MainLayout.set_minimap_image(new TextureRegionDrawable(ngol.getBoard()));
				time_delayed = 0;
			}
		}

		if(Gdx.input.isKeyPressed(Input.Keys.MINUS)){
			zoom_value -= 0.1f;
		}
		if(Gdx.input.isKeyPressed(Input.Keys.PLUS)){
			zoom_value += 0.1f;
		}
		if(1.0f > zoom_value)
			zoom_value = 1.0f;

		Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
		stage.act(Gdx.graphics.getDeltaTime());

		stage.getBatch().begin();
		stage.getBatch().draw(
			ngol.getBoard(),
				0,//(Gdx.graphics.getWidth()*zoom_value - Gdx.graphics.getWidth())/2.0f,
				0,//(Gdx.graphics.getHeight()*zoom_value - Gdx.graphics.getHeight())/2.0f,
			Gdx.graphics.getWidth()*zoom_value,
			Gdx.graphics.getHeight()*zoom_value
		);
		stage.getBatch().end();

		stage.draw();
	}
	
	@Override
	public void dispose () {
		stage.dispose();
	}
}
