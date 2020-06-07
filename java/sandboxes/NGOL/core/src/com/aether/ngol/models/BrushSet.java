package com.aether.ngol.models;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.Pixmap;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.scenes.scene2d.ui.*;
import com.badlogic.gdx.scenes.scene2d.utils.Drawable;
import com.badlogic.gdx.scenes.scene2d.utils.TextureRegionDrawable;

import java.util.ArrayList;

public class BrushSet extends Table {
    Skin used_skin;
    ButtonGroup<ImageButton> brush_button_group;
    ArrayList<Pixmap> brushes;

    public BrushSet(Skin skin){
        super(skin);
        used_skin = skin;
        top().left();
        brush_button_group = new ButtonGroup<>();
        brush_button_group.setMinCheckCount(0);
        brush_button_group.setMaxCheckCount(1);

        brushes = new ArrayList<>();
    }

    private ImageButton.ImageButtonStyle get_brush_style(){
        ImageButton.ImageButtonStyle style = new ImageButton.ImageButtonStyle();
        style.imageChecked = used_skin.getDrawable("brush_selected");
        return style;
    }

    private void refresh(){
        clear();
        for(ImageButton brush : brush_button_group.getButtons())
            add(brush).expand().fill().minSize(32).maxSize(64).row();
    }

    private Pixmap remove_corners_of(Pixmap image){
        image.setBlending(Pixmap.Blending.None);
        Vector2 middle = new Vector2(image.getWidth()/2.0f,image.getHeight()/2.0f);
        for(int x = 0; x < image.getWidth(); ++x){
            for(int y = 0; y < image.getWidth(); ++y){
                if(new Vector2(x,y).dst(middle) > (image.getWidth()/2.1f)){
                    image.drawPixel(x,y, Color.rgba8888(0.0f,0.0f,0.0f,0.0f));
                }
            }
        }
        return image;
    }

    public void addBrush(Texture texture){
        if(!texture.getTextureData().isPrepared())
            texture.getTextureData().prepare();
        Pixmap tex = texture.getTextureData().consumePixmap();
        brushes.add(tex);

        ImageButton.ImageButtonStyle style = get_brush_style();
        style.up = new TextureRegionDrawable(new Texture(remove_corners_of(tex)));
        ImageButton new_brush = new ImageButton(style);
        brush_button_group.add(new_brush);
        refresh();
    }

    public void select_last_brush(){
        select_brush(brush_button_group.getButtons().size - 1);
    }

    public void select_brush(int index) {
        if((0 <= index)&&(brush_button_group.getButtons().size > index)){
            brush_button_group.getButtons().get(index).setChecked(true);
        }
    }

    public void remove_brush(int index){
        if(0 < brush_button_group.getButtons().size){
            if((0 < index) && (brush_button_group.getButtons().size > index)){ /* Remove Brush at index */
                brushes.remove(index);
                brush_button_group.remove(brush_button_group.getButtons().get(index));
            }else if(-1 != brush_button_group.getCheckedIndex()){ /* Remove selected brush */
                brushes.remove(brush_button_group.getCheckedIndex());
                brush_button_group.remove(brush_button_group.getButtons().get(brush_button_group.getCheckedIndex()));
            }else{ /* Remove last Brush */
                brushes.remove(brushes.size()-1);
                brush_button_group.remove(brush_button_group.getButtons().get(brush_button_group.getButtons().size-1));
            }
            refresh();
        }
    }

    public Pixmap get_selected_brush(){
        if(-1 != brush_button_group.getCheckedIndex())
            return brushes.get(brush_button_group.getCheckedIndex());
        else return null;
    }

    public void update_selected_brush(Pixmap new_brush){
        if(-1 != brush_button_group.getCheckedIndex()) {
            brushes.set(brush_button_group.getCheckedIndex(), new_brush);
            brush_button_group.getButtons().get(brush_button_group.getCheckedIndex()).getStyle().up = (
                new TextureRegionDrawable( new Texture(
                       remove_corners_of(brushes.get(brush_button_group.getCheckedIndex()))
                ))
            );
        }
    }

}
