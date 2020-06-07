package com.aether.ngol.models;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.scenes.scene2d.InputEvent;
import com.badlogic.gdx.scenes.scene2d.ui.Image;
import com.badlogic.gdx.scenes.scene2d.ui.Skin;
import com.badlogic.gdx.scenes.scene2d.ui.WidgetGroup;
import com.badlogic.gdx.scenes.scene2d.utils.DragListener;
import com.badlogic.gdx.scenes.scene2d.utils.TextureRegionDrawable;

public class Minimap extends WidgetGroup {
    private Image minimap;
    private Image minimap_border;
    private Image minimap_position;

    private float zoom_width_value = 2.0f;
    private float screen_ratio;
    private Vector2 dimensions;

    Minimap(Skin used_skin, Vector2 dimensions_){
        setTransform(false);
        dimensions = dimensions_;
        minimap_border = new Image(used_skin.getRegion("minimap"));
        minimap_position = new Image(used_skin.getRegion("minimap_position"));
        minimap = new Image();

        minimap.setFillParent(true);
        minimap_border.setFillParent(true);

        minimap_position.setPosition(0,0);
        minimap_position.addListener(new DragListener(){
            float prevX;
            float prevY;

            @Override
            public boolean touchDown(InputEvent event, float x, float y, int pointer, int button) {
                System.out.println("touchDown!");
                prevX = x;
                prevY = y;
                return super.touchDown(event, x, y, pointer, button);
            }

            @Override
            public void touchDragged(InputEvent event, float x, float y, int pointer) {
                step((x-prevX),(y-prevY));
                prevX = x;
                prevY = y;
                super.touchDragged(event, x, y, pointer);
            }
        });

        layout();
        addActor(minimap);
        addActor(minimap_border);
        addActor(minimap_position);
    }

    public Vector2 get_world_size(){
        return new Vector2(
        (getWidth() * Gdx.graphics.getWidth())/minimap_position.getWidth(),
        (getHeight() * Gdx.graphics.getHeight())/minimap_position.getHeight()
        );
    }

    public float get_zoom(){
        return zoom_width_value;
    }

    public Vector2 get_camera_position(){
        return new Vector2(
        (minimap_position.getX() * Gdx.graphics.getWidth())/minimap_position.getWidth(),
        (minimap_position.getY() * Gdx.graphics.getHeight())/minimap_position.getHeight()
        );
    }

    public  Vector2 get_render_coordinates(Vector2 mouse_coords){
        return new Vector2(
        ((get_camera_position().x + mouse_coords.x) * dimensions.x / get_world_size().x),
        ((get_camera_position().y + mouse_coords.y) * dimensions.y / get_world_size().y)
        );
    }

    public void adjust_zoom(float val){
        if(
            (getWidth() * screen_ratio / (zoom_width_value + val)) <= getWidth()
        ){
            zoom_width_value += val;
            if(1.0f > zoom_width_value)
                zoom_width_value = 1.0f;
            layout();
        }
    }

    @Override
    public void layout() {
        setHeight(getWidth() * (dimensions.x / dimensions.y));
        screen_ratio = ((float)Gdx.graphics.getWidth() / Gdx.graphics.getHeight());
        minimap_position.setSize(
            getWidth() * screen_ratio / zoom_width_value,
            getHeight() / zoom_width_value
        );

    }

    public void set_map_image(TextureRegion minimap_image){
        minimap.setDrawable(new TextureRegionDrawable(minimap_image));
        layout();
    }

    public void step(float x, float y){
        Vector2 addition = new Vector2(x , y );
        Vector2 new_position = new Vector2(minimap_position.getX(),minimap_position.getY()).add(addition);
        if((0 > new_position.x)||((new_position.x + minimap_position.getWidth()) > minimap_border.getWidth())){
            addition.x = 0.00001f;
        }
        if((0 > new_position.y)||((new_position.y + minimap_position.getHeight()) > minimap_border.getHeight())){
            addition.y = 0.00001f;
        }
        minimap_position.moveBy(addition.x, addition.y);
    }
}
