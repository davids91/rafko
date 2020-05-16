package com.crystalline.aether;

import com.badlogic.gdx.ApplicationAdapter;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.crystalline.aether.models.NGOL;
import com.crystalline.aether.models.Reality;

import java.util.Random;

public class GameClass extends ApplicationAdapter {
	SpriteBatch batch;
	Texture img_aether;
	Texture img_nether;

	OrthographicCamera camera;
	ShapeRenderer shapeRenderer;
	BitmapFont font;
	Reality world;
	NGOL ngol;

	final int[] world_block_number = {25,25};
	final float world_block_size = 100.0f;
	final float[] world_size = {world_block_number[0] * world_block_size, world_block_number[1] * world_block_size};

	final Random rnd = new Random();

	@Override
	public void create () {
		Gdx.gl.glClearColor(0, 0.1f, 0.1f, 1);
		camera = new OrthographicCamera(world_block_number[0] * world_block_size, world_block_number[1] * world_block_size);
		camera.translate(camera.viewportWidth/2.0f, camera.viewportHeight/2.0f);
		camera.update();

		batch = new SpriteBatch();
		shapeRenderer = new ShapeRenderer();
		img_aether = new Texture("aether.png");
		img_nether = new Texture("nether.png");
		font = new BitmapFont();

		world = new Reality(world_block_number[0], world_block_number[1]);
		ngol = new NGOL(
				world_block_number[0],
				world_block_number[1],
				4f, 9f
		);
	}

	private void drawblock(int x, int y, float scale_, Texture tex){
		float scale = Math.max( -1.0f, Math.min( 1.0f , scale_) );
		batch.setColor(world.aether_value_at(x,y),world.ratio_at(x,y),world.nether_value_at(x,y),1.0f);
		batch.draw(
			tex,
			x * world_block_size + (world_block_size/2.0f) - (world_block_size/2.0f) * Math.abs(scale),
			y * world_block_size + (world_block_size/2.0f) - (world_block_size/2.0f) * Math.abs(scale),
			(Math.abs(scale) * world_block_size),
			(Math.abs(scale) * world_block_size)
		);
		font.draw(
			batch,
			String.format( "Ae: %.2f", world.aether_value_at(x,y) ),
			x * world_block_size + (world_block_size/4.0f),
			y * world_block_size + (3.0f * world_block_size/4.0f)
		);
		font.draw(
			batch,
			String.format( "Ne: %.2f", world.nether_value_at(x,y) ),
			x * world_block_size + (world_block_size/4.0f),
			y * world_block_size + (world_block_size/2.0f)
		);
		font.draw(
			batch,
			String.format( "R: %.2f", world.ratio_at(x,y) ),
			x * world_block_size + (world_block_size/4.0f),
			y * world_block_size + (world_block_size/4.0f)
		);
	}

	private void drawGrid(float lineWidth, float cellSize) {
		shapeRenderer.begin(ShapeRenderer.ShapeType.Filled);
		shapeRenderer.setColor(Color.BROWN);
		for(float x = cellSize;x<world_size[0];x+=cellSize){
			shapeRenderer.rect(x,0,lineWidth,world_size[1]);
		}
		for(float y = cellSize;y<world_size[0];y+=cellSize){
			shapeRenderer.rect(0,y,world_size[0],lineWidth);
		}
		shapeRenderer.end();
	}

	public void my_game_loop(){
		world.main_loop(0.01f);
		if(Gdx.input.isKeyJustPressed(Input.Keys.TAB)){
			ngol.loop();
		}
		if(Gdx.input.isKeyPressed(Input.Keys.SPACE)){
			ngol.loop();
		}
		if(Gdx.input.isKeyJustPressed(Input.Keys.ENTER)){
			world.randomize();
			ngol.reset(1.0f,1.0f,1.0f);
		}
	}

	@Override
	public void render () {
		my_game_loop();

		Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
		batch.begin();
		batch.setProjectionMatrix(camera.combined);
//		for(int x = 0; x < world_block_number[0]; ++x){
//			for(int y = 0; y < world_block_number[1]; ++y){
//				drawblock(x,y, world.aether_value_at(x,y), img_aether);
//				drawblock(x,y, world.nether_value_at(x,y), img_nether);
//			}
//		}
		batch.draw(new Texture(ngol.getBoard()),0,0,world_size[0],world_size[1]);
		batch.end();
		shapeRenderer.setProjectionMatrix(camera.combined);
		drawGrid(1.0f, world_block_size);
	}
	
	@Override
	public void dispose () {
		batch.dispose();
	}
}
