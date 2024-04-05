# -*- coding: utf-8 -*-
"""
Created on Fri Apr 5 19:26:11 2024

@author: Debabrata Ghorai, Ph.D.

Flask Application - Manage PyGeoML Projects.
"""

import os
import sys
sys.path.append('src')

from flask import Flask, request, render_template, jsonify

# app = Flask(__name__)
app = Flask(__name__, template_folder='templates')



app_id = 1


match app_id:
    case 1:
        from regression.random_forest.pipeline.prediction_pipeline import CustomData, PredictPipeline
        @app.route('/')
        def home_page():
            return render_template('/regression/random_forest/index.html')


        @app.route('/predict', methods=['GET', 'POST'])
        def predict_user_data():
            if request.method == 'GET':
                return render_template('/regression/random_forest/form.html')
            else:
                user_inputs = {
                    'value1': float(request.form['crim']),
                    'value2': float(request.form['zn']),
                    'value3': float(request.form['indus']),
                    'value4': float(request.form['chas']),
                    'value5': float(request.form['age']),
                    'value6': float(request.form['dis']),
                    'value7': float(request.form['rad']),
                    'value8': float(request.form['b']),
                    'value9': float(request.form['lstat'])
                }
                user_data = CustomData(**user_inputs)
                y_test = user_data.get_user_inputs()
                predict_pipeline = PredictPipeline()
                res = predict_pipeline.predict(y_test)
                results = round(res[0], 2)
                return render_template('/regression/random_forest/results.html', final_result=results)
    case 2:
        from classification.xgboost.pipeline.prediction_pipeline import CustomData, PredictPipeline
        @app.route('/')
        def home_page():
            return render_template('/classification/xgboost/index.html')
        
        @app.route('/predict', methods=['GET', 'POST'])
        def predict_user_data():
            if request.method == 'GET':
                return render_template('/classification/xgboost/form.html')
            else:
                user_inputs = {
                    'age': float(request.form['age']),
                    'workclass': str(request.form['workclass']),
                    'fnlwgt': float(request.form['fnlwgt']),
                    'education': str(request.form['education']),
                    'education_num': float(request.form['education_num']),
                    'marital_status': str(request.form['marital_status']),
                    'occupation': str(request.form['occupation']),
                    'relationship': str(request.form['relationship']),
                    'race': str(request.form['race']),
                    'sex': str(request.form['sex']),
                    'capital_gain': float(request.form['capital_gain']),
                    'capital_loss': float(request.form['capital_loss']),
                    'hours_per_week': float(request.form['hours_per_week']),
                    'native_country': str(request.form['native_country'])
                }
                user_data = CustomData(**user_inputs)
                y_test = user_data.get_user_inputs()
                predict_pipeline = PredictPipeline()
                results = predict_pipeline.predict(y_test)
                return render_template('/classification/xgboost/results.html', final_result=results)
            


if __name__ == '__main__':
    app.run(debug=True)
