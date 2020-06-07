package com.aether.ngol;

import com.aether.ngol.models.MainLayout;
import com.aether.ngol.services.NGOL;
import com.badlogic.gdx.*;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.math.Vector3;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.EventListener;
import com.badlogic.gdx.scenes.scene2d.InputEvent;
import com.badlogic.gdx.scenes.scene2d.ui.*;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;
import com.badlogic.gdx.scenes.scene2d.utils.DragListener;

import java.util.HashMap;

public class NGOL_Main extends ApplicationAdapter {
	private MainLayout main_layout;

	NGOL ngol;
	float time_delayed = 0;

	@Override
	public void create () {
		HashMap<String, EventListener> actions = new HashMap<>();
		actions.put("reset",new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				ngol.randomize();
			}
		});
		actions.put("step",new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				ngol.loop();
				main_layout.getMinimap().set_map_image(ngol.getBoard());
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
		actions.put("startCapture", new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				main_layout.startCapture();
			}
		});
		actions.put("touch",new DragListener(){
			@Override
			public void touchDragged(InputEvent event, float x, float y, int pointer) {
				if(null != main_layout.getBrushPanel().get_selected_brush()){
					if(null != main_layout.getBrushPanel().get_selected_brush())
						ngol.addBrush(
								new Texture(main_layout.getBrushPanel().get_selected_brush()),
								main_layout.getMinimap().get_render_coordinates(new Vector2(
										main_layout.getStage().getCamera().unproject(new Vector3(Gdx.input.getX(), Gdx.input.getY(),0)).x,
										main_layout.getStage().getCamera().unproject(new Vector3(Gdx.input.getX(), Gdx.input.getY(),0)).y
								)),
								new Vector2(256/main_layout.getMinimap().get_zoom(),256/main_layout.getMinimap().get_zoom())
						);
				}
				super.touchDragged(event, x, y, pointer);
			}
		});

		HashMap<String,Float> data = new HashMap<>();
		data.put("ngol-width", 2048.0f);
		data.put("ngol-height", 2048.0f);
		main_layout = new MainLayout(actions,data);
		main_layout.setInputProcessor();

		Gdx.gl.glClearColor(0.2f, 0.5f, 0.1f, 1);
		ngol = new NGOL(2048,2048, 2f, 3f);
		ngol.randomize();
		main_layout.getMinimap().set_map_image(ngol.getBoard());
	}

	@Override
	public void resize(int width, int height){
		main_layout.layout();
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

		if(Gdx.input.isKeyPressed(Input.Keys.MINUS)){
			main_layout.getMinimap().adjust_zoom(-0.05f);
		}
		if(Gdx.input.isKeyPressed(Input.Keys.PLUS)){
			main_layout.getMinimap().adjust_zoom(+0.05f);
		}
		if(Gdx.input.isKeyPressed(Input.Keys.LEFT)){
			main_layout.getMinimap().step(-10,0);
		}
		if(Gdx.input.isKeyPressed(Input.Keys.RIGHT)){
			main_layout.getMinimap().step(10,0);
		}
		if(Gdx.input.isKeyPressed(Input.Keys.UP)){
			main_layout.getMinimap().step(0,10);
		}
		if(Gdx.input.isKeyPressed(Input.Keys.DOWN)){
			main_layout.getMinimap().step(0,-10);
		}

		Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
		main_layout.getStage().act(Gdx.graphics.getDeltaTime());

		main_layout.getStage().getBatch().begin();
		main_layout.getStage().getBatch().draw(
			ngol.getBoard(),
			-main_layout.getMinimap().get_camera_position().x,
			-main_layout.getMinimap().get_camera_position().y,
			main_layout.getMinimap().get_world_size().x,
			main_layout.getMinimap().get_world_size().y
		);
		main_layout.getStage().getBatch().end();

		main_layout.getStage().draw();
	}
	
	@Override
	public void dispose () {
		main_layout.getStage().dispose();
	}
}
