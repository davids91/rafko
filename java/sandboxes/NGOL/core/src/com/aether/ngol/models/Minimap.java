package com.aether.ngol.models;

import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.scenes.scene2d.InputEvent;
import com.badlogic.gdx.scenes.scene2d.ui.Image;
import com.badlogic.gdx.scenes.scene2d.ui.Skin;
import com.badlogic.gdx.scenes.scene2d.ui.WidgetGroup;
import com.badlogic.gdx.scenes.scene2d.utils.DragListener;
import com.badlogic.gdx.scenes.scene2d.utils.Drawable;
import com.badlogic.gdx.scenes.scene2d.utils.TextureRegionDrawable;

public class Minimap extends WidgetGroup{
    private Image minimap;
    private Image minimap_border;
    private Image minimap_position;

    private Vector2 dimensions;
    private float zoom_value = 1.0f;

    Minimap(Skin used_skin, Vector2 dimensions_){
        dimensions = dimensions_;
        minimap_border = new Image(used_skin.getRegion("panel"));
        minimap_border.setFillParent(true);
        minimap = new Image();
        minimap_position = new Image(used_skin.getRegion("minimap_position"));
        minimap_position.addListener(new DragListener(){
            public void drag(InputEvent event, float x, float y, int pointer) {
                Vector2 addition = new Vector2(x - minimap_position.getWidth() / 2, y - minimap_position.getHeight() / 2);
                Vector2 new_position = new Vector2(minimap_position.getX(),minimap_position.getY()).add(addition);
                if((minimap_border.getWidth() * 0.025f > new_position.x)||((new_position.x + minimap_position.getWidth()) > (minimap_border.getWidth() - minimap_border.getWidth() * 0.025f))){
                    addition.x = 0;
                }
                if((minimap_border.getHeight() * 0.025f > new_position.y)||((new_position.y + minimap_position.getHeight()) > (minimap_border.getHeight() - minimap_border.getHeight() * 0.025f))){
                    addition.y = 0;
                }

                minimap_position.moveBy(addition.x, addition.y);
            }
        });

        addActor(minimap_border);
        addActor(minimap);
        addActor(minimap_position);
    }

    public Vector2 get_position(Vector2 size){
        return get_position(size.x, size.y);
    }

    public Vector2 get_position(float width, float height){
        return new Vector2(
            (((minimap_position.getX() - minimap_border.getWidth() * 0.025f) * width * zoom_value) / minimap.getWidth()),
            (((minimap_position.getY() - minimap_border.getHeight() * 0.025f) * height * zoom_value) / minimap.getHeight())
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
        minimap_border.setSize(getWidth(),getHeight());
        minimap.setSize(minimap_border.getWidth() * 0.95f, minimap_border.getHeight() * 0.95f);
        minimap.setPosition(minimap_border.getWidth() * 0.025f, minimap_border.getHeight() * 0.025f);
        minimap_position.setSize(minimap.getWidth() / zoom_value, minimap.getHeight() / zoom_value);
    }

    public void set_map_image(TextureRegion minimap_image){
        minimap.setDrawable(new TextureRegionDrawable(minimap_image));
        dimensions.x = minimap_image.getRegionWidth();
        dimensions.x = minimap_image.getRegionHeight();
        minimap_position.setSize(minimap.getWidth(), minimap.getHeight());
        minimap_position.setPosition(minimap_border.getWidth() * 0.025f, minimap_border.getHeight() * 0.025f);
        layout();
    }
}
