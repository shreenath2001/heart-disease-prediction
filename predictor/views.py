from django.shortcuts import render

import pandas as pd

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

import joblib

model = joblib.load(open("model/grid_search_rf.sav", "rb"))

# Create your views here.
def home(request):
    return render(request, 'predictor/home.html')

def result(request):
    features = {}
    message = "Sorry, unable to predict."
    if request.method == 'POST':
        features['age'] = int(request.POST['age'])
        features['sex'] = int(request.POST['sex'])
        features['cp'] = int(request.POST['cp'])
        features['trestbps'] = int(request.POST['trestbps'])
        features['chol'] = int(request.POST['chol'])
        features['fbs'] = int(request.POST['fbs'])
        features['restecg'] = int(request.POST['restecg'])
        features['thalach'] = int(request.POST['thalach'])
        features['exang'] = int(request.POST['exang'])
        features['oldpeak'] = float(request.POST['oldpeak'])
        features['slope'] = int(request.POST['slope'])
        features['ca'] = int(request.POST['ca'])

        features['sex_0'] = 0
        features['sex_1'] = 0
        features['cp_0'] = 0
        features['cp_1'] = 0
        features['cp_2'] = 0
        features['cp_3'] = 0
        features['fbs_0'] = 0
        features['fbs_1'] = 0
        features['restecg_0'] = 0
        features['restecg_1'] = 0
        features['restecg_2'] = 0
        features['exang_0'] = 0
        features['exang_1'] = 0
        features['slope_0'] = 0
        features['slope_1'] = 0
        features['slope_2'] = 0
        features['ca_0'] = 0
        features['ca_1'] = 0
        features['ca_2'] = 0
        features['ca_3'] = 0
        features['ca_4'] = 0

        if(features['sex'] == 0):
            features['sex_0'] = 1
        else:
            features['sex_1'] = 1

        if(features['cp'] == 0):
            features['cp_0'] = 1
        elif(features['cp'] == 1):
            features['cp_1'] = 1
        elif(features['cp'] == 2):
            features['cp_2'] = 1
        else:
            features['cp_3'] = 1
        
        if(features['fbs'] == 0):
            features['fbs_0'] = 1
        else:
            features['fbs_1'] = 1

        if(features['restecg'] == 0):
            features['restecg_0'] = 1
        elif(features['restecg'] == 1):
            features['restecg_1'] = 1
        else:
            features['restecg_2'] = 1

        if(features['exang'] == 0):
            features['exang_0'] = 1
        else:
            features['exang_1'] = 1

        if(features['slope'] == 0):
            features['slope_0'] = 1
        elif(features['slope'] == 1):
            features['slope_1'] = 1
        else:
            features['slope_2'] = 1

        if(features['ca'] == 0):
            features['ca_0'] = 1
        elif(features['ca'] == 1):
            features['ca_1'] = 1
        elif(features['ca'] == 2):
            features['ca_2'] = 1
        elif(features['ca'] == 3):
            features['ca_3'] = 1
        else:
            features['ca_4'] = 1

        df = pd.DataFrame(features, index  =[0])
        df.drop(labels=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca'], axis=1, inplace=True)
        prediction = model.predict(df)
        if(prediction[0] == 0):
            message = "You are free from heart disease."
        elif(prediction[0] == 1):
            message = "You might be suffering from a heart disease."
        data = {
            'message':message,
        }

        return render(request, 'predictor/result.html', data)
