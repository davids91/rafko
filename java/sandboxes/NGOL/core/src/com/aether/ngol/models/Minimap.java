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

public class Minimap extends WidgetGroup{
    private Image minimap;
    private Image minimap_border;
    private Image minimap_position;

    private float zoom_value = 0.5f;
    private Vector2 dimensions;

    Minimap(Skin used_skin, Vector2 dimensions_){
        dimensions = dimensions_;
        minimap_border = new Image(used_skin.getRegion("minimap_position"));
        minimap_border.setFillParent(true);
        minimap = new Image();
        minimap_position = new Image(used_skin.getRegion("minimap_position"));
        minimap_position.setSize(minimap.getWidth() * zoom_value, minimap.getHeight() * zoom_value);
        minimap_position.addListener(new DragListener(){
            public void drag(InputEvent event, float x, float y, int pointer) {
                Vector2 addition = new Vector2(x - minimap_position.getWidth() / 2, y - minimap_position.getHeight() / 2);
                Vector2 new_position = new Vector2(minimap_position.getX(),minimap_position.getY()).add(addition);
                if((minimap_border.getWidth() * 0.025f > new_position.x)||((new_position.x + minimap_position.getWidth()) > (minimap_border.getWidth() - minimap_border.getWidth() * 0.025f))){
                    addition.x = 0.00001f;
                }
                if((minimap_border.getHeight() * 0.025f > new_position.y)||((new_position.y + minimap_position.getHeight()) > (minimap_border.getHeight() - minimap_border.getHeight() * 0.025f))){
                    addition.y = 0.00001f;
                }

                minimap_position.moveBy(addition.x, addition.y);
            }
        });
        minimap_position.setPosition(0,0);

        addActor(minimap);
        addActor(minimap_border);
        addActor(minimap_position);
    }

    public Vector2 get_position(){
        return new Vector2(
            ((minimap_position.getX() * Gdx.graphics.getWidth()*get_zoom()) / minimap.getWidth()),
            ((minimap_position.getY() * Gdx.graphics.getHeight()*get_zoom()) / minimap.getHeight())
        );
    }
    public Vector2 get_world_coordinates(Vector2 mouse_coords){
        return new Vector2(
                get_position().x + (mouse_coords.x * Gdx.graphics.getWidth() / (dimensions.x * get_zoom())),
                get_position().y + (mouse_coords.y * Gdx.graphics.getHeight() / (dimensions.y * get_zoom()))
        );
    }

    public float get_zoom(){
        return zoom_value;
    }

    public void adjust_zoom(float val){
        zoom_value += val;
        if(1.0f > zoom_value)
            zoom_value = 1.0f;
        layout();
    }

    @Override
    public void layout() {
        setHeight(getWidth() * (Gdx.graphics.getWidth() / Gdx.graphics.getHeight()));
        minimap_border.setSize(getWidth(),getHeight());
        minimap.setSize(getWidth(),getHeight());
        minimap_position.setSize(
            minimap.getWidth() / zoom_value,
            minimap.getHeight() / zoom_value
        );
    }

    public void set_map_image(TextureRegion minimap_image){
        minimap.setDrawable(new TextureRegionDrawable(minimap_image));
        minimap_position.setSize(minimap.getWidth(), minimap.getHeight());
        layout();
    }
}
