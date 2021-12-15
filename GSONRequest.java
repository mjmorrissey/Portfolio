package com.company;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.net.URL;



public class GSONRequest {

//initialize variables that will be need in AnyLogic as public static

    public static BlockStorage blockStorage; //create public BlockStorage object
    static private HttpURLConnection connection;

    public void Get() {


        // Method 1: java.net.HttpURLConnection
        BufferedReader reader;
        String line;
        StringBuffer responseContent = new StringBuffer();
        try {
            //URL url = new URL("https://jsonplaceholder.typicode.com/albums");
            URL url = new URL("http://127.0.0.1:5000/");
            connection = (HttpURLConnection) url.openConnection();

            //request setup
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(5000);
            connection.setReadTimeout(5000);

            int status = connection.getResponseCode();
            //System.out.println(status);
            if (status > 299) {
                reader = new BufferedReader(new InputStreamReader(connection.getErrorStream()));
                while ((line = reader.readLine()) != null) {
                    responseContent.append(line);
                }
                reader.close();
            } else {
                reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));
                while ((line = reader.readLine()) != null) {
                    responseContent.append(line);
                }
                reader.close();
                parse(responseContent.toString()); //apply parse function
            }
            //System.out.println(responseContent.toString());
        } catch (MalformedURLException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            connection.disconnect();
        }
    }

    public static String parse (String responseBody){ //json string is input
        Gson gson = new Gson(); //new gson object
        blockStorage = gson.fromJson(responseBody, BlockStorage.class); //use fromJson() to turn string into blockStorage object
        return null;
    }


    public BlockStorage get_class() {
        return blockStorage;
    }

}


