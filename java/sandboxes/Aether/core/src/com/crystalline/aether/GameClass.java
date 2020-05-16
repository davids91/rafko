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
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.crystalline.aether.models.Elemental_plane;
import com.crystalline.aether.models.NGOL;
import com.crystalline.aether.models.Nethereal_plane;

public class GameClass extends ApplicationAdapter {
	SpriteBatch batch;
	Texture img_aether;
	Texture img_nether;

	OrthographicCamera camera;
	ShapeRenderer shapeRenderer;
	BitmapFont font;
	Nethereal_plane nethereal_plane;
	Elemental_plane elemental_plane;
	NGOL ngol;

	final int[] world_block_number = {15,15};
	final float world_block_size = 100.0f;
	final float[] world_size = {world_block_number[0] * world_block_size, world_block_number[1] * world_block_size};
	my_rule_type my_rule;
	boolean rule_set = false;

	public class my_rule_type implements NGOL.NGOL_RULE {

		@Override
		public Color execute(Color pixel, float proxR, float proxG, float proxB, int x, int y) {
			return new Color(
					Math.min(1.0f, nethereal_plane.aether_value_at(x,y)/10.0f),
					pixel.g,//Math.min(1.0f,world.ratio_at(x,y)/10.0f),
					Math.min(1.0f, nethereal_plane.nether_value_at(x,y)/10.0f),
				1.0f
			);
		}
	};

	@Override
	public void create () {
		Gdx.gl.glClearColor(0, 0.1f, 0.1f, 1);
		camera = new OrthographicCamera();
		camera.setToOrtho(false,world_block_number[0] * world_block_size, world_block_number[1] * world_block_size);
		camera.update();

		batch = new SpriteBatch();
		shapeRenderer = new ShapeRenderer();
		img_aether = new Texture("aether.png");
		img_nether = new Texture("nether.png");
		font = new BitmapFont();

		nethereal_plane = new Nethereal_plane(world_block_number[0], world_block_number[1]);
		elemental_plane = new Elemental_plane(world_block_number[0], world_block_number[1]);
		elemental_plane.pond_with_grill((int)(world_block_number[1]/2.0f));
		nethereal_plane.attach_to(elemental_plane);
		ngol = new NGOL(
				world_block_number[0],
				world_block_number[1],
				4f, 9f
		);
		my_rule = new my_rule_type();
	}

	private void drawblock(int x, int y, float scale_, Texture tex){
		float scale = Math.max( -1.0f, Math.min( 1.0f , scale_) );
		batch.setColor(nethereal_plane.aether_value_at(x,y), nethereal_plane.ratio_at(x,y), nethereal_plane.nether_value_at(x,y),0.5f);
		batch.draw(
			tex,
			x * world_block_size + (world_block_size/2.0f) - (world_block_size/2.0f) * Math.abs(scale),
			y * world_block_size + (world_block_size/2.0f) - (world_block_size/2.0f) * Math.abs(scale),
			(Math.abs(scale) * world_block_size),
			(Math.abs(scale) * world_block_size)
		);
		font.draw(
			batch,
			String.format( "Ae: %.2f", nethereal_plane.aether_value_at(x,y) ),
			x * world_block_size + (world_block_size/4.0f),
			y * world_block_size + (3.0f * world_block_size/4.0f)
		);
		font.draw(
			batch,
			String.format( "Ne: %.2f", nethereal_plane.nether_value_at(x,y) ),
			x * world_block_size + (world_block_size/4.0f),
			y * world_block_size + (world_block_size/2.0f)
		);
		font.draw(
			batch,
			String.format( "R: %.2f", nethereal_plane.ratio_at(x,y) ),
			x * world_block_size + (world_block_size/4.0f),
			y * world_block_size + (world_block_size/4.0f)
		);
	}

	private void drawGrid(float lineWidth, float cellSize) {
		shapeRenderer.begin(ShapeRenderer.ShapeType.Filled);
		shapeRenderer.setColor(Color.LIME);
		for(float x = cellSize;x<world_size[0];x+=cellSize){
			shapeRenderer.rect(x,0,lineWidth,world_size[1]);
		}
		for(float y = cellSize;y<world_size[0];y+=cellSize){
			shapeRenderer.rect(0,y,world_size[0],lineWidth);
		}
		shapeRenderer.end();
	}

	public void my_game_loop(){
		nethereal_plane.main_loop(0.01f);
		ngol.loop();
		if(Gdx.input.isKeyJustPressed(Input.Keys.TAB)){
			if(!rule_set){
				ngol.setRule(my_rule);
				rule_set = true;
			}else{
				ngol.setRule(null);
				rule_set = false;
			}
		}
		if(Gdx.input.isKeyPressed(Input.Keys.ENTER)){
			nethereal_plane.add_nether_to(world_block_number[0]/2,world_block_number[1]/2 + 1, 0.5f);
		}
//		if(Gdx.input.isKeyPressed(Input.Keys.TAB)){
//			nethereal_plane.main_loop(0.01f);
//			ngol.loop();
//		}
		if(Gdx.input.isKeyJustPressed(Input.Keys.BACKSPACE)){
			nethereal_plane.attach_to(elemental_plane);
			ngol.reset(1.0f,1.0f,1.0f);
		}
	}

	@Override
	public void render () {
		my_game_loop();

		Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
		batch.begin();
		batch.setProjectionMatrix(camera.combined);
		TextureRegion lofaszbazdmeg = new TextureRegion(new Texture(elemental_plane.getWorldImage()));
		lofaszbazdmeg.flip(false,true);
		batch.draw(lofaszbazdmeg,0,0,world_size[0],world_size[1]);
//		lofaszbazdmeg = new TextureRegion(new Texture(ngol.getBoard()));
//		lofaszbazdmeg.flip(false,true);
//		batch.draw(lofaszbazdmeg,0,0,world_size[0],world_size[1]);
		for(int x = 0; x < world_block_number[0]; ++x){
			for(int y = 0; y < world_block_number[1]; ++y){
				drawblock(x,y, nethereal_plane.aether_value_at(x,y), img_aether);
				drawblock(x,y, nethereal_plane.nether_value_at(x,y), img_nether);
			}
		}
		batch.end();
		shapeRenderer.setProjectionMatrix(camera.combined);
		drawGrid(1.0f, world_block_size);
	}
	
	@Override
	public void dispose () {
		batch.dispose();
	}
}
