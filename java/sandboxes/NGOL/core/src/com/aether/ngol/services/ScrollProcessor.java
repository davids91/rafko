package com.aether.ngol.services;

import com.badlogic.gdx.InputProcessor;

public class ScrollProcessor implements InputProcessor {

    public interface My_scroll_action_interface {
        void scrollAction(int scrollValue);
    };

    private My_scroll_action_interface my_action;

    public ScrollProcessor(My_scroll_action_interface fnc){
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
        return false;
    }

    @Override
    public boolean touchUp(int screenX, int screenY, int pointer, int button) {
        return false;
    }

    @Override
    public boolean touchDragged(int screenX, int screenY, int pointer) {
        return false;
    }

    @Override
    public boolean mouseMoved(int screenX, int screenY) {
        return false;
    }

    @Override
    public boolean scrolled(int amount) {
        my_action.scrollAction(amount);
        return false;
    }
}
