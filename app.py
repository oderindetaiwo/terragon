import json
import pickle
import numpy as np
from flask import Flask, request
# 

flask_app = Flask(__name__)

#ML model path
model_path = "ML_Model/model.pkl"


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
        age = request.form.get('age')
        bs_fast = request.form.get('BS_Fast')
        bs_pp = request.form.get('BS_pp')
        plasma_r = request.form.get('Plasma_R')
        plasma_f = request.form.get('Plasma_F')
        HbA1c = request.form.get('HbA1c')
        fields = [age,bs_fast,bs_pp,plasma_r,plasma_f,HbA1c]
        if not None in fields:
            #Datapreprocessing Convert the values to float
            age = float(age)
            bs_fast = float(bs_fast)
            bs_pp = float(bs_pp)
            plasma_r = float(plasma_r)
            plasma_f = float(plasma_f)
            hbA1c = float(HbA1c)
            result = [age,bs_fast,bs_pp,plasma_r,plasma_f,HbA1c]
            #Passing data to model & loading the model from disk
            classifier = pickle.load(open(model_path, 'rb'))
            prediction = classifier.predict([result])[0]
            conf_score =  np.max(classifier.predict_proba([result]))*100
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
    flask_app.run(host ='0.0.0.0',port=8080, debug=False)
ML_Model
The ML_Model directory contains the ML model, the data I used to train the model and the pickle file generated after model is being trained which the API will make use of.

requirements.txt
The requirements.txt file is a text file which contains all the required python packages we need for our application to run. Some of the packages I made of use were:
Flask==1.1.2
pandas==1.0.3
numpy==1.18.2
sklearn==0.0
Dockerfile
A Dockerfile is a text file that defines a Docker image. You'll use a Dockerfile to create your own custom Docker image when the base image you want to use for your project doesn't meet your required needs. For the model I'll be deploying, this is how my Dockefile looks like:
#I specify the parent base image which is the python version 3.7
FROM python:3.7

MAINTAINER aminu israel <aminuisrael2@gmail.com>

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
    && apt-get -y install gcc make \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --no-cache-dir --upgrade pip

# set work directory
WORKDIR /src/app

# copy requirements.txt
COPY ./requirements.txt /src/app/requirements.txt

# install project requirements
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# Generate pikle file
WORKDIR /src/app/ML_Model
RUN python model.py

# set work directory
WORKDIR /src/app

# set app port
EXPOSE 8080

ENTRYPOINT [ "python" ] 

# Run app.py when the container launches
CMD [ "app.py","run","--host","0.0.0.0"] 