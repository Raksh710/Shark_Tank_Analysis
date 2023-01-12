#!/usr/bin/env python
# coding: utf-8

# In[3]:


# importing all the neccessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, render_template, request
import jsonify
import requests

app = Flask(__name__)
#model = pickle.load(open('Final_Shark_Tank_Pipeline.pkl', 'rb'))
model = joblib.load('pipeline_knn_44_final.pkl')

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
   # Fuel_Type_Diesel=0
    if request.method == 'POST':
        Number_of_Presenters = int(request.form['Number_of_Presenters'])
        Male1=int(request.form['Male1'])
        Male2=int(request.form['Male2'])
        Male3=int(request.form['Male3'])
        Male4=int(request.form['Male4'])
        Novelties = int(request.form['Novelty'])
        Health_Wellness=int(request.form['Health/Wellness'])
        Food_and_Beverage=int(request.form['Food_and_Beverage'])
        Business_Services = int(request.form['Business_Services'])
        Software_Tech=int(request.form['Software/Tech'])
        Children_Education=int(request.form['Children/Education'])
        Lifestyle_Home=int(request.form['Lifestyle/Home'])
        Automotive=int(request.form['Automotive'])
        Fashion_Beauty = int(request.form['Fashion/Beauty'])
        Media_Entertainment = int(request.form['Media/Entertainment'])
        Fitness_Sports_Outdoor = int(request.form['Fitness/Sports/Outdoor'])
        Pet_Products = int(request.form['Pet_Products']) 
        Travel = int(request.form['Travel'])
        Green_CleanTech=int(request.form['Green/CleanTech'])
        Uncertain_Other=int(request.form['Uncertain/Other'])
        MalePresenter=int(request.form['MalePresenter'])
        FemalePresenter=int(request.form['FemalePresenter'])
        MixedGenderPresenters = int(request.form['MixedGenderPresenters'])
        AmountRequested=float(request.form['AmountRequested'])
        EquityRequested=float(request.form['EquityRequested'])
        ImpliedValuationRequested=float(request.form['ImpliedValuationRequested'])
        BarbaraCorcoran=int(request.form['BarbaraCorcoran'])
        MarkCuban=int(request.form['MarkCuban'])
        LoriGreiner=int(request.form['LoriGreiner'])
        RobertHerjavec=int(request.form['RobertHerjavec'])
        DaymondJohn = int(request.form['DaymondJohn'])
        KevinOLeary = int(request.form['KevinOLeary'])
        KevinHarrington=int(request.form['KevinHarrington'])
        Guest=int(request.form['Guest'])
        Total_Males=int(request.form['Total_Males']) 
        Total_Females=int(request.form['Total_Females'])
        Total_Pitchers = int(request.form['Total_Pitchers'])
        
        Region = request.form['Region']
        if (Region == "East North Central"):
            Region_East_North_Central = 1
            Region_East_South_Central = 0
            Region_Mid_Atlantic = 0
            Region_Mountain = 0
            Region_New_England = 0
            Region_Pacific = 0
            Region_South_Atlantic = 0
            Region_West_North_Central = 0
            Region_West_South_Central = 0
            
        elif (Region == "East South Central"):
            Region_East_North_Central = 0
            Region_East_South_Central = 1
            Region_Mid_Atlantic = 0
            Region_Mountain = 0
            Region_New_England = 0
            Region_Pacific = 0
            Region_South_Atlantic = 0
            Region_West_North_Central = 0
            Region_West_South_Central = 0
            
        elif (Region == "Mid-Atlantic"):
            Region_East_North_Central = 0
            Region_East_South_Central = 0
            Region_Mid_Atlantic = 1
            Region_Mountain = 0
            Region_New_England = 0
            Region_Pacific = 0
            Region_South_Atlantic = 0
            Region_West_North_Central = 0
            Region_West_South_Central = 0
            
        elif (Region == "Mountain"):
            Region_East_North_Central = 0
            Region_East_South_Central = 0
            Region_Mid_Atlantic = 0
            Region_Mountain = 1
            Region_New_England = 0
            Region_Pacific = 0
            Region_South_Atlantic = 0
            Region_West_North_Central = 0
            Region_West_South_Central = 0
        
        elif (Region == "New England"):
            Region_East_North_Central = 0
            Region_East_South_Central = 0
            Region_Mid_Atlantic = 0
            Region_Mountain = 0
            Region_New_England = 1
            Region_Pacific = 0
            Region_South_Atlantic = 0
            Region_West_North_Central = 0
            Region_West_South_Central = 0
        
        elif (Region == "Pacific"):
            
            Region_East_North_Central = 0
            Region_East_South_Central = 0
            Region_Mid_Atlantic = 0
            Region_Mountain = 0
            Region_New_England = 0
            Region_Pacific = 1
            Region_South_Atlantic = 0
            Region_West_North_Central = 0
            Region_West_South_Central = 0
            
        elif (Region == "South Atlantic"):
            Region_East_North_Central = 0
            Region_East_South_Central = 0
            Region_Mid_Atlantic = 0
            Region_Mountain = 0
            Region_New_England = 0
            Region_Pacific = 0
            Region_South_Atlantic = 1
            Region_West_North_Central = 0
            Region_West_South_Central = 0
            
        elif (Region == "West North Central"):
            
            Region_East_North_Central = 0
            Region_East_South_Central = 0
            Region_Mid_Atlantic = 0
            Region_Mountain = 0
            Region_New_England = 0
            Region_Pacific = 0
            Region_South_Atlantic = 0
            Region_West_North_Central = 1
            Region_West_South_Central = 0
              
        elif (Region == "West South Central"):
            Region_East_North_Central = 0
            Region_East_South_Central = 0
            Region_Mid_Atlantic = 0
            Region_Mountain = 0
            Region_New_England = 0
            Region_Pacific = 0
            Region_South_Atlantic = 0
            Region_West_North_Central = 0
            Region_West_South_Central = 1
        
        Eth1 = int(request.form['Eth1'])
        if (Eth1==0):
            Eth1_0 = 1
            Eth1_1 = 0
            Eth1_2 = 0
            Eth1_3 = 0
            Eth1_4 = 0
            
        elif (Eth1==1):
            Eth1_0 = 0
            Eth1_1 = 1
            Eth1_2 = 0
            Eth1_3 = 0
            Eth1_4 = 0
            
        elif (Eth1==2):
            Eth1_0 = 0
            Eth1_1 = 0
            Eth1_2 = 1
            Eth1_3 = 0
            Eth1_4 = 0
            
        elif (Eth1==3):
            Eth1_0 = 0
            Eth1_1 = 0
            Eth1_2 = 0
            Eth1_3 = 1
            Eth1_4 = 0
        elif (Eth1==4):
            Eth1_0 = 0
            Eth1_1 = 0
            Eth1_2 = 0
            Eth1_3 = 0
            Eth1_4 = 1
       
        Eth2 = int(request.form['Eth2'])
        if (Eth2==0):
            Eth2_0 = 1
            Eth2_1 = 0
            Eth2_2 = 0
            Eth2_3 = 0
            Eth2_4 = 0
            
        elif (Eth2==1):
            Eth2_0 = 0
            Eth2_1 = 1
            Eth2_2 = 0
            Eth2_3 = 0
            Eth2_4 = 0
            
        elif (Eth2==2):
            Eth2_0 = 0
            Eth2_1 = 0
            Eth2_2 = 1
            Eth2_3 = 0
            Eth2_4 = 0
            
        elif (Eth2==3):
            Eth2_0 = 0
            Eth2_1 = 0
            Eth2_2 = 0
            Eth2_3 = 1
            Eth2_4 = 0
        elif (Eth2==4):
            Eth2_0 = 0
            Eth2_1 = 0
            Eth2_2 = 0
            Eth2_3 = 0
            Eth2_4 = 1
       
        Eth3 = int(request.form['Eth3'])
        if (Eth3==0):
            Eth3_0 = 1
            Eth3_1 = 0
            Eth3_2 = 0
            Eth3_4 = 0
            
        elif (Eth3==1):
            Eth3_0 = 0
            Eth3_1 = 1
            Eth3_2 = 0
            Eth3_4 = 0
            
        elif (Eth3==2):
            Eth3_0 = 0
            Eth3_1 = 0
            Eth3_2 = 1
            Eth3_4 = 0
            
        elif (Eth3==4):
            Eth3_0 = 0
            Eth3_1 = 0
            Eth3_2 = 0
            Eth3_4 = 1
            
        Eth4 = int(request.form['Eth4'])
        if (Eth4==0):
            Eth4_0 = 0
            Eth4_2 = 0
            
            
        elif (Eth4==2):
            Eth4_0 = 0
            Eth4_2 = 2
          
        Eth5 = int(request.form['Eth5'])
        if (Eth5==0):
            Eth5_0 = 0
            Eth5_2 = 0
            
        elif (Eth5==2):
            Eth5_0 = 0
            Eth5_2 = 2
        
        pred = model.predict([[Number_of_Presenters, Male1, Male2, Male3, Male4, Novelties,
                                   Health_Wellness, Food_and_Beverage, Business_Services,
                                   Lifestyle_Home, Software_Tech , Children_Education,
                                   Automotive, Fashion_Beauty , Media_Entertainment,
                                   Fitness_Sports_Outdoor, Pet_Products, Travel,
                                   Green_CleanTech, Uncertain_Other, MalePresenter,
                                   FemalePresenter, MixedGenderPresenters, AmountRequested,
                                   EquityRequested, ImpliedValuationRequested, BarbaraCorcoran,
                                   MarkCuban, LoriGreiner, RobertHerjavec, DaymondJohn,
                                   KevinOLeary, KevinHarrington, Guest, Total_Males,
                                   Total_Females, Total_Pitchers, Region_East_North_Central,
                                   Region_East_South_Central, Region_Mid_Atlantic, Region_Mountain,
                                   Region_New_England, Region_Pacific, Region_South_Atlantic,
                                   Region_West_North_Central, Region_West_South_Central, Eth1_0,
                                   Eth1_1 , Eth1_2, Eth1_3, Eth1_4, Eth2_0, Eth2_1, Eth2_2,
                                   Eth2_3, Eth2_4, Eth3_0, Eth3_1, Eth3_2, Eth3_4, Eth4_0,
                                   Eth4_2, Eth5_0, Eth5_2 ]])
                           
        deal_yes_prob = model.predict_proba([[Number_of_Presenters, Male1, Male2, Male3, Male4, Novelties,
                                   Health_Wellness, Food_and_Beverage, Business_Services,
                                   Lifestyle_Home, Software_Tech , Children_Education,
                                   Automotive, Fashion_Beauty , Media_Entertainment,
                                   Fitness_Sports_Outdoor, Pet_Products, Travel,
                                   Green_CleanTech, Uncertain_Other, MalePresenter,
                                   FemalePresenter, MixedGenderPresenters, AmountRequested,
                                   EquityRequested, ImpliedValuationRequested, BarbaraCorcoran,
                                   MarkCuban, LoriGreiner, RobertHerjavec, DaymondJohn,
                                   KevinOLeary, KevinHarrington, Guest, Total_Males,
                                   Total_Females, Total_Pitchers, Region_East_North_Central,
                                   Region_East_South_Central, Region_Mid_Atlantic, Region_Mountain,
                                   Region_New_England, Region_Pacific, Region_South_Atlantic,
                                   Region_West_North_Central, Region_West_South_Central, Eth1_0,
                                   Eth1_1 , Eth1_2, Eth1_3, Eth1_4, Eth2_0, Eth2_1, Eth2_2,
                                   Eth2_3, Eth2_4, Eth3_0, Eth3_1, Eth3_2, Eth3_4, Eth4_0,
                                   Eth4_2, Eth5_0, Eth5_2 ]])
        #prediction=model.predict([[Present_Price,Kms_Driven2,Owner,Year,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Mannual]])
        output= pred[0]
        yes_deal_prob = round(100*deal_yes_prob[0][1] , 2)
        
        if output==0:
            return render_template('index.html',prediction_texts=f"No Deal. Your chances of getting a deal is only: {yes_deal_prob}%. Sorry, but we wish you all the best.")
        else:
            return render_template('index.html',prediction_text=f"You got the deal. Strong probabilty of {yes_deal_prob}% of getting the deal.")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True,use_reloader=False)

