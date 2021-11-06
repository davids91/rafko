package models;

public class ServerSlot_data {
    String id = "";
    public ServerSlot_data(String id_){
        id = id_;
    }

    public String getName(){
        return id;
    }
}
