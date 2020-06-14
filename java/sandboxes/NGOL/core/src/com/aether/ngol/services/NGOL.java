package com.aether.ngol.services;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.*;
import com.badlogic.gdx.graphics.g2d.Batch;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.graphics.glutils.FrameBuffer;
import com.badlogic.gdx.graphics.glutils.ShaderProgram;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.utils.ScreenUtils;

import java.util.Random;


/**
 * NGOL meaning "Not Game Of Life" is an adaptation of the original game of life,
 * using summaries and thresholds analougous to the original games.
 */
public class NGOL {

    Batch batch;
    FrameBuffer[] ngolBuffer;
    int usedBuf = 0;
    Vector2 board_size;

    String vertex_shader;
    String randomize_shader;
    String main_loop_shader;
    ShaderProgram reset_program;
    ShaderProgram main_program;

    /**
     * Interesting ranges:
     * - 0.7 ; 1.0 - Square trip
     * - 1.0 ; 1.0 - Trip
     * - 2.0 ; 3.0 - Game Of Life ??
     * - 1.9 ; 2.9 - Game Of Life !!
     * - 2.5 ; 4.0 - New Game of life 2 ??
     * - 3.0 ; 5.0 - Building Squares
     * - 3.0 ; 7.0 - New Game of Life??
     * - 4.0 ; 9.0 - Stable land
     */
    float underPopThr = 4f;
    float overPopThr = 9f;

    final Random my_random = new Random();
    TextureRegion placeholder_texture;

    public NGOL(int width, int height, float uThr, float oThr) {
        board_size = new Vector2(width, height);
        batch = new SpriteBatch();
        OrthographicCamera ngol_view = new OrthographicCamera(width, height);
        ngol_view.translate(board_size.x/2,board_size.y/2);
        ngol_view.update();
        batch.setProjectionMatrix(ngol_view.combined);

        Gdx.gl.glClearColor(0.0f, 0.0f, 0.0f, 1);
        placeholder_texture = new TextureRegion(new Texture(width, height, Pixmap.Format.RGBA8888));
        placeholder_texture.getTexture().setFilter(Texture.TextureFilter.Nearest, Texture.TextureFilter.Nearest);

        ngolBuffer = new FrameBuffer[2];
        ngolBuffer[0] = new FrameBuffer(Pixmap.Format.RGBA8888, width, height, false);
        ngolBuffer[0].getColorBufferTexture().setFilter(Texture.TextureFilter.Nearest, Texture.TextureFilter.Nearest);
        ngolBuffer[0].begin();
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
        ngolBuffer[0].end();

        ngolBuffer[1] = new FrameBuffer(Pixmap.Format.RGBA8888, width, height, false);
        ngolBuffer[1].getColorBufferTexture().setFilter(Texture.TextureFilter.Nearest, Texture.TextureFilter.Nearest);
        ngolBuffer[1].begin();
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
        ngolBuffer[1].end();

        underPopThr = uThr;
        overPopThr = oThr;

        ShaderProgram.pedantic = false;
        vertex_shader = Gdx.files.internal("shaders/vanilla.vshr").readString();
        randomize_shader = Gdx.files.internal("shaders/ngol_randomize.fshr").readString();

        reset_program = new ShaderProgram(vertex_shader, randomize_shader);
        if (reset_program.getLog().length()!=0)
            System.out.println(reset_program.getLog());

        load_main_shader("");
    }

    public void load_main_shader(String custom_part){
        main_loop_shader = Gdx.files.internal("shaders/ngol_main.fshr").readString();

        if (!reset_program.isCompiled()) {
            System.err.println(reset_program.getLog());
            System.exit(0);
        }

        main_loop_shader = main_loop_shader.replace("$CUSTOM_RULE$",custom_part);
        main_program = new ShaderProgram(vertex_shader, main_loop_shader);
        if (main_program.getLog().length()!=0)
            System.out.println(main_program.getLog());

        if (!main_program.isCompiled()) {
            System.err.println(main_program.getLog());
        }
        /*!Note: This might come in handy ==> shader_program.setUniform3fv("my_data", my_float_array, 0, my_float_array.length); */
        setThresholds(underPopThr,overPopThr);
    }

    public void addBrush(Texture tex, Vector2 position){
        tex.setFilter(Texture.TextureFilter.Nearest, Texture.TextureFilter.MipMapNearestNearest);
        batch.setShader(null);
        ngolBuffer[usedBuf].begin();
        batch.begin();
        batch.draw(tex,(position.x - tex.getWidth()/2.0f),(position.y - tex.getHeight()/2.0f), tex.getWidth(),tex.getHeight());
        batch.end();
        ngolBuffer[usedBuf].end();
    }

    public Pixmap flipPixmap(Pixmap src) {
        final int width = src.getWidth();
        final int height = src.getHeight();
        Pixmap flipped = new Pixmap(width, height, src.getFormat());

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                flipped.drawPixel(x, y, src.getPixel(x, height - y - 1));
            }
        }
        return flipped;
    }

    public Texture get_brush(Vector2 position, Vector2 size){
        ngolBuffer[usedBuf].begin();
        batch.begin();
        Texture tex = new Texture(flipPixmap(
                ScreenUtils.getFrameBufferPixmap((int)(position.x - size.x/2.0f),(int)(position.y - size.y/2.0f), (int)size.x,(int)size.y)
        ));
        batch.end();
        ngolBuffer[usedBuf].end();
        return tex;
    }

    public void loop(){
        /* bind main shader */
        batch.setShader(main_program);

        /* Send required variables */
        ngolBuffer[usedBuf].getColorBufferTexture().bind(0);
        main_program.setUniformi("u_texture",0);

        /* bind in-active BFO */
        ngolBuffer[(usedBuf + 1)%2].begin();
        batch.begin();

        /* render main program */
        main_program.setUniformf("my_seed", my_random.nextFloat());
        batch.draw(placeholder_texture,0,0,board_size.x,board_size.y);

        /* unbind stuff */
        batch.end();
        ngolBuffer[(usedBuf + 1)%2].end();

        usedBuf = (usedBuf + 1)%2;
    }

    public void randomize(){ randomize(1.0f,1.0f,1.0f); }
    public void randomize(float red_intensity, float green_intensity, float blue_intensity){
        usedBuf = 0;
        /* bind reset shader */
        batch.setShader(reset_program);

        /* bind active BFO */
        ngolBuffer[usedBuf].begin();
        batch.begin();
        Gdx.gl.glClearColor(0.0f, 1.0f, 0.0f, 1);

        /* Send required variables */
        reset_program.setUniformf("my_seed", my_random.nextFloat());
        reset_program.setUniformf("red_intensity", red_intensity);
        reset_program.setUniformf("green_intensity", green_intensity);
        reset_program.setUniformf("blue_intensity", blue_intensity);

        /* render randomize program */
        Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
        batch.draw(placeholder_texture,0,0,board_size.x,board_size.y);

        /* unbind stuff */
        batch.end();
        ngolBuffer[usedBuf].end();
    }

    public void setThresholds(float uThr, float oThr){
        setUnderPopThr(uThr);
        setOverPopThr(oThr);
    }

    public void setUnderPopThr(float uThr){
        underPopThr = uThr;
        main_program.begin();
        main_program.setUniformf("underPopThr", underPopThr);
        main_program.end();
    }
    public void setOverPopThr(float oThr){
        overPopThr = oThr;
        main_program.begin();
        main_program.setUniformf("overPopThr", overPopThr);
        main_program.end();
    }
    public TextureRegion getBoard(){
        placeholder_texture = new TextureRegion(ngolBuffer[usedBuf].getColorBufferTexture());
        placeholder_texture.flip(false,true);
        return placeholder_texture;
    }

    public Vector2 getSize(){
        return board_size;
    }

}
