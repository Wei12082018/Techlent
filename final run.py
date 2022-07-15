#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify, render_template
import json
import numpy as np
import pickle
import joblib

model = joblib.load('final_model.pkl')


# Create Flask App
app = Flask(__name__)

# Create API routing call
@app.route('/', methods= ['GET','POST'])
def predict():
    pred = ''
    X = []
    if request.method == 'POST':
        age_range = request.form['age_range']
        marital_status = request.form['marital_status']
        rented = request.form['rented']
        family_size = request.form['family_size']
        no_of_children = request.form['no_of_children']
        income_bracket = request.form['income_bracket']
        cust_quantity = request.form["cust_quantity"]
        cust_other_usage = request.form['cust_other_usage']
        cust_coupon_usage = request.form['cust_coupon_usage']
        camp_type = request.form['camp_type']
        camp_start_month = request.form['camp_start_month']
        camp_end_month = request.form['camp_end_month']
        camp_duration = request.form['camp_duration']
        holiday_covered = request.form['holiday_covered']
        item_NO = request.form['item_NO']
        brand_NO = request.form['brand_NO']
        items_quantity = request.form['items_quantity']
        Grocery = request.form['Grocery']
        Pharmaceutical = request.form['Pharmaceutical']
        Food = request.form['Food']
        Drink = request.form['Drink']
        Miscellaneous = request.form['Miscellaneous']
        Natural_Products = request.form['Natural Products']
        Garden = request.form['Garden']
        Skin_Hair_Care = request.form['Skin & Hair Care']
        X = np.array([[str(age_range),
                       str(marital_status),
                       str(rented),
                       str(family_size),
                       str(no_of_children),
                       str(income_bracket),
                       float(cust_quantity), 
                       float(cust_other_usage), 
                       float(cust_coupon_usage),
                       str(camp_type),
                       int(camp_start_month),
                       int(camp_end_month),
                       float(camp_duration),
                       str(holiday_covered),
                       float(item_NO),
                       float(brand_NO),
                       float(items_quantity),
                       int(Grocery),
                       int(Pharmaceutical),
                       int(Food),
                       int(Drink),
                       int(Miscellaneous),
                       int(Natural_Products),
                       int(Garden),
                       int(Skin_Hair_Care)]])
        pred = model.predict(X)
        
    return render_template('index.html', pred=pred, X=X)


if __name__ == '__main__':
    # Load Model and Feature columns
    model = joblib.load('final_model.pkl')
    col_names = joblib.load('column_names.pkl')
    app.run(debug=True, host='127.0.0.1', port=5000)

