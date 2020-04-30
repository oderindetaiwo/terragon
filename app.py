import json
import pickle
import numpy as np
import pandas as pd
import time
from flask import Flask, request, jsonify
from sklearn.externals import joblib


flask_app = Flask(__name__)

#ML model path
model_path = "model/model.pkl"
column_path = "model/model_columns,pkl"


@flask_app.route('/', methods=['GET'])
def index_page():
    return_data = {
        "error" : "0",
        "message" : "Successful"
    }
    return flask_app.response_class(response=json.dumps(return_data), mimetype='application/json')

@flask_app.route('/predict',methods=['GET'])
def model_deploy():
    try: 
        data_dict = {'location_region': request.form.get('location_region'),
        'location_state': request.form.get('location_state'),
        'device_manufacturer': request.form.get('device_manufacturer'),
        'spend_total': request.form.get('spend_total'),
        'spend_vas': request.form.get('spend_vas'),
        'spend_voice': request.form.get('spend_voice'),
        'spend_data': request.form.get('spend_data'),
        'sms_cost': request.form.get('sms_cost'),
        'xtra_data_talk_rev': request.form.get('xtra_data_talk_rev'),
        'customer_class': request.form.get('customer_class'),
        'customer_value': request.form.get('customer_value'),
        'location_city': request.form.get('location_city'),
        'gender': request.form.get('gender'),
        'age': request.form.get('age'),
        'device_type': request.form.get('device_type')}

        data= list(data_dict.items())
        data = (pd.DataFrame(data)).T
        data.columns = data.iloc[0]
        data = data.iloc[1:]
        
        if not None in data:
            #Datapreprocessing Convert the values to float
            cat_list = ['location_region', 'location_state','customer_value','gender','device_type','device_manufacturer']
            for a in cat_list:
                data[a].fillna('unspecified',inplace=True)
                
                
            num_list = ['spend_total', 'spend_vas', 'spend_voice', 'spend_data','sms_cost','xtra_data_talk_rev', 'customer_class','age']
            for b in num_list:
                data[b].fillna(value=0, inplace=True)


            for j in data['device_manufacturer']:
                if j == 'tecno':
                    j = j
                elif j == 'itel':
                    j= j
                elif j == 'infinix':
                    j= j
                elif j == 'samsung':
                    j= j
                elif j == 'nokia':
                    j= j
                elif j == 'apple':
                    j= j
                else:
                    data['device_manufacturer'].replace(j,'others',inplace=True)


            for l in data['customer_value']:
                if l == 'low' :
                    data['customer_value'].replace(l,1,inplace=True)
                elif l == 'medium':
                    data['customer_value'].replace(l,2,inplace=True)
                elif l == 'high' :
                    data['customer_value'].replace(l,3,inplace=True)
                elif l == 'very high' :
                    data['customer_value'].replace(l,4,inplace=True)
                elif l == 'top' :
                    data['customer_value'].replace(l,5,inplace=True)
                else:
                    data['customer_value'].replace(l,0,inplace=True)


            data = pd.get_dummies(data, columns=cat_list, dummy_na=True)
            
            model_columns =pickle.load(open(column_path, 'rb'))
            data = data.reindex(columns = model_columns, fill_value= 0)

            data = data.iloc[0]
            #Passing data to model & loading the model from disk
            rf = pickle.load(open(model_path, 'rb'))
            prediction = rf.predict([data])[0]
            conf_score =  np.max(rf.predict_proba([data]))*100
            return_data = {
                "error" : '0',
                "message" : 'Successfull',
                "prediction": prediction,
                "confidence_score" : conf_score
            }
        else:
            return_data = {
                "error" : '1',
                "message": "Invalid Parameters"             
            }
    except Exception as e:
        return_data = {
            'error' : '2',
            "message": str(e)
            }
    return flask_app.response_class(response=json.dumps(return_data), mimetype='application/json')

if __name__ == "__main__":
    flask_app.run(host ='0.0.0.0',port=3400, debug=False)
