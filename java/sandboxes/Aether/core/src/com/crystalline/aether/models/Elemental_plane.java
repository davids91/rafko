package com.crystalline.aether.models;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.Pixmap;

public class Elemental_plane  extends Reality_plane{
    Materials.names[][] blocks;

    public Elemental_plane(int sizeX_, int sizeY_){
        super(sizeX_,sizeY_);
        blocks = new Materials.names[sizeX][sizeY];
        for(int x = 0;x < sizeX; ++x){
            for(int y = 0; y < sizeY; ++y){
                blocks[x][y] = Materials.names.Air;
            }
        }
    }

    /**
     * Create a simple pond with some fire on one side
     * @param floorHeight - the height of the ground floor
     */
    public void pond_with_grill(int floorHeight){
        /* create the ground floor */
        for(int x = 0;x < sizeX; ++x){
            for(int y = 0; y < sizeY; ++y){
                if(y <= floorHeight)blocks[x][y] = Materials.names.Earth;
                    else blocks[x][y] = Materials.names.Air;
            }
        }

        /* Create the pond */
        int posX; int posY;
        for(float radius = 0.0f; radius < (floorHeight/2.0f); radius += 1.0f){
            for(float sector = (float)Math.PI * 0.9f; sector <= Math.PI * 2.1f; sector += Math.PI / 180.0f){
                posX = (int)(sizeX/2.0f + (float)Math.cos(sector) * radius);
                posY = (int)(floorHeight + (float)Math.sin(sector) * radius);
                blocks[posX][posY] = Materials.names.Water;
            }
        }

        /* Create some fire */
        posX = (int)(sizeX/2.0f - floorHeight/2.0f) - 2;
        posY = floorHeight + 1;
        blocks[posX][posY] = Materials.names.Fire;
        blocks[posX-1][posY] = Materials.names.Fire;
        blocks[posX+1][posY] = Materials.names.Fire;
        blocks[posX][posY+1] = Materials.names.Fire;
    }

    public Materials.names element_at(int x, int y){
        return blocks[x][y];
    }

    public Pixmap getWorldImage(){
        Pixmap worldImage = new Pixmap(sizeX,sizeY, Pixmap.Format.RGB888);
        for(int x = 0;x < sizeX; ++x){
            for(int y = 0; y < sizeY; ++y){
                worldImage.drawPixel(
                    x,y, Color.rgba8888(Materials.colors[blocks[x][y].ordinal()])
                );
            }
        }
        return worldImage;
    }

}
