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
    ButtonGroup<ImageButton> brush_buttons;
    ArrayList<Pixmap> brushes;

    public BrushSet(Skin skin){
        super(skin);
        used_skin = skin;
        top().left();
        brush_buttons = new ButtonGroup<>();
        brush_buttons.setMinCheckCount(0);
        brush_buttons.setMaxCheckCount(1);

        brushes = new ArrayList<>();
    }

    private ImageButton.ImageButtonStyle get_brush_style(){
        ImageButton.ImageButtonStyle style = new ImageButton.ImageButtonStyle();
        style.imageChecked = used_skin.getDrawable("brush_selected");
        return style;
    }

    private void refresh(){
        clear();
        for(ImageButton brush : brush_buttons.getButtons())
            add(brush).expand().fill().minSize(32).maxSize(64).row();
    }

    private Drawable remove_corners_of(Pixmap image){
        image.setBlending(Pixmap.Blending.None);
        Vector2 middle = new Vector2(image.getWidth()/2.0f,image.getHeight()/2.0f);
        for(int x = 0; x < image.getWidth(); ++x){
            for(int y = 0; y < image.getWidth(); ++y){
                if(new Vector2(x,y).dst(middle) > (image.getWidth()/2.1f)){
                    image.drawPixel(x,y, Color.rgba8888(0.0f,0.0f,0.0f,0.0f));
                }
            }
        }
        return new TextureRegionDrawable(new Texture(image));
    }

    public void addBrush(Texture texture){
        if(!texture.getTextureData().isPrepared())
            texture.getTextureData().prepare();
        Pixmap tex = texture.getTextureData().consumePixmap();
        brushes.add(tex);

        ImageButton.ImageButtonStyle style = get_brush_style();
        style.up = remove_corners_of(tex);
        ImageButton new_brush = new ImageButton(style);
        brush_buttons.add(new_brush);
        refresh();
    }

    public void removeBrush(int index){
        if(0 < brush_buttons.getButtons().size){
            if((0 < index) && (brush_buttons.getButtons().size > index)){ /* Remove Brush at index */
                brushes.remove(index);
                brush_buttons.remove(brush_buttons.getButtons().get(index));
            }else if(-1 != brush_buttons.getCheckedIndex()){ /* Remove selected brush */
                brushes.remove(brush_buttons.getCheckedIndex());
                brush_buttons.remove(brush_buttons.getButtons().get(brush_buttons.getCheckedIndex()));
            }else{ /* Remove last Brush */
                brushes.remove(brushes.size()-1);
                brush_buttons.remove(brush_buttons.getButtons().get(brush_buttons.getButtons().size-1));
            }
            refresh();
        }
    }

    public Pixmap get_selected_brush(){
        if(-1 != brush_buttons.getCheckedIndex())
            return brushes.get(brush_buttons.getCheckedIndex());
        else return null;
    }

}
