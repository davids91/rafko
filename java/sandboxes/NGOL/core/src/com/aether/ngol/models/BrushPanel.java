package com.aether.ngol.models;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.Pixmap;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.TextureRegion;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.ui.*;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;
import com.badlogic.gdx.scenes.scene2d.utils.TextureRegionDrawable;
import com.badlogic.gdx.utils.Align;


public class BrushPanel extends Table {
    Button brushes[];
    HorizontalGroup buttons_panel;
    ImageButton load_button;
    ImageButton save_button;
    ImageButton capture_button;
    TextButton add_button;
    TextButton remove_button;
    BrushSet currentSet;

    public BrushPanel(Skin used_skin){
        BitmapFont bitmapFont = new BitmapFont(Gdx.files.internal("font-export.fnt"), used_skin.getRegion("font-export"));
        setBackground(used_skin.getDrawable("panel"));
        top().left().padLeft(Value.percentWidth(.25f,this));

        ImageButton.ImageButtonStyle imageButtonStyle_load = new ImageButton.ImageButtonStyle();
        imageButtonStyle_load.up = used_skin.getDrawable("button");
        imageButtonStyle_load.down = used_skin.getDrawable("button-pressed");
        imageButtonStyle_load.imageUp = used_skin.getDrawable("icon-folder");
        load_button = new ImageButton(imageButtonStyle_load);
        load_button.getImage().setScale(0.7f,0.7f);
        load_button.getImage().setOrigin(Align.center);

        ImageButton.ImageButtonStyle imageButtonStyle_save = new ImageButton.ImageButtonStyle();
        imageButtonStyle_save.up = used_skin.getDrawable("button");
        imageButtonStyle_save.down = used_skin.getDrawable("button-pressed");
        imageButtonStyle_save.imageUp = used_skin.getDrawable("icon-save");
        save_button = new ImageButton(imageButtonStyle_save);
        save_button.getImage().setScale(0.6f,0.6f);
        save_button.getImage().setOrigin(Align.center);

        ImageButton.ImageButtonStyle imageButtonStyle_capture = new ImageButton.ImageButtonStyle();
        imageButtonStyle_capture.up = used_skin.getDrawable("button");
        imageButtonStyle_capture.down = used_skin.getDrawable("button-pressed");
        imageButtonStyle_capture.imageUp = used_skin.getDrawable("icon-file-image");
        capture_button = new ImageButton(imageButtonStyle_capture);
        capture_button.setTransform(true);
        capture_button.setScale(1.0f,0.95f);

        TextButton.TextButtonStyle textButtonStyle = new TextButton.TextButtonStyle();
        textButtonStyle.font = bitmapFont;
        textButtonStyle.up = used_skin.getDrawable("button");
        textButtonStyle.down = used_skin.getDrawable("button-pressed");

        add_button = new TextButton("+",textButtonStyle);
        add_button.setTransform(true);
        add_button.setOrigin(Align.left | Align.top);
        add_button.setScale(0.8f);
        add_button.addListener(new ChangeListener() {
            @Override
            public void changed(ChangeEvent event, Actor actor) {
                currentSet.addBrush(new Texture(Gdx.files.internal("brush.png"), Pixmap.Format.RGBA8888,true));
            }
        });
        remove_button = new TextButton("-",textButtonStyle);
        remove_button.setTransform(true);
        remove_button.setOrigin(Align.right | Align.top);
        remove_button.setScale(0.8f);
        remove_button.addListener(new ChangeListener() {
            @Override
            public void changed(ChangeEvent event, Actor actor) {
                currentSet.removeBrush(-1);
            }
        });

        Table file_op_group = new Table();
        Table brush_op_group = new Table().center();
        file_op_group.add(load_button);
        file_op_group.add(save_button);
        brush_op_group.add(remove_button).right();
        brush_op_group.add(capture_button);
        brush_op_group.add(add_button).left();
        add(file_op_group).row();
        add(brush_op_group);
        add(buttons_panel).row();
        currentSet = new BrushSet(used_skin);
        add(currentSet).expandY().fill();
    }

    public Pixmap get_selected_brush(){
        return currentSet.get_selected_brush();
    }
}
