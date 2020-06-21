package com.aether.ngol.services;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.InputProcessor;

public class MouseInputProcessor implements InputProcessor {

    public interface My_scroll_action_interface {
        boolean scrollAction(int scrollValue);
        boolean mouseMoveAction(int screenX, int screenY);
        boolean touchDownAction(int screenX, int screenY, int pointer, int button);
    };

    private My_scroll_action_interface my_action;

    public MouseInputProcessor(My_scroll_action_interface fnc){
        my_action = fnc;
    }

    @Override
    public boolean keyDown(int keycode) {
        return false;
    }

    @Override
    public boolean keyUp(int keycode) {
        return false;
    }

    @Override
    public boolean keyTyped(char character) {
        return false;
    }

    @Override
    public boolean touchDown(int screenX, int screenY, int pointer, int button) {
        return my_action.touchDownAction(screenX,screenY,pointer,button);
    }

    @Override
    public boolean touchUp(int screenX, int screenY, int pointer, int button) {
        return false;
    }

    @Override
    public boolean touchDragged(int screenX, int screenY, int pointer){
        return my_action.touchDownAction(screenX,screenY,pointer,0);
    }

    @Override
    public boolean mouseMoved(int screenX, int screenY) {
        return my_action.mouseMoveAction(screenX, Gdx.graphics.getHeight() - screenY);
    }

    @Override
    public boolean scrolled(int amount) {
        return my_action.scrollAction(amount);
    }
}
