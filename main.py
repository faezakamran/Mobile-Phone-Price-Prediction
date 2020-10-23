from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import pandas as pd
import numpy as np
import sklearn
from category_encoders import *

app = Flask(__name__)
model = pickle.load(open('mobile_phone_price_prediction_xgb_model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')
    
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        brand = str(request.form['brand'])
        acer_model = str(request.form['acer-model'])
        alcatel_model = str(request.form['alcatel-model'])
        pcamera = str(request.form['primary-camera'])
        scamera = str(request.form['secondary-camera'])
        cpu = str(request.form['CPU'])
        ram = str(request.form['RAM'])
        os = str(request.form['OS'])
        pmodel = ""

        if acer_model == "select-acer-model":
            pmodel = alcatel_model  
        else:
            pmodel = acer_model

        data_df = get_data(brand, pmodel, pcamera, scamera, cpu, ram, os)
        encoded_data_df = encode_categorical_data(data_df)
        
        prediction=model.predict(encoded_data_df)
        output=round(prediction[0],2)
        print(prediction)

        return render_template('index.html',prediction_text="Predicted price is €".format(output))
    else:
        return render_template('index.html')
        

def get_data(br, md, pc, sc, cpu, ram, os):
    df = pd.read_csv("preprocessed_data.csv") 

    new_df = df.query('brand == "%s"' % br and 'model == "%s"' % md and 'primary_camera == "%s"' % pc and 'secondary_camera == "%s"' % sc and 'CPU == "%s"' % cpu and 'RAM == "%s"' % ram and 'OS == "%s"' % os)
    
    return new_df

def encode_categorical_data(new_features):
    
    enc = OrdinalEncoder().fit(new_features)
    encoded_features = enc.transform(new_features)
    
    #encoded_features = pd.get_dummies(new_features)
    
    return encoded_features

if __name__=="__main__":
    app.run(debug=True)
