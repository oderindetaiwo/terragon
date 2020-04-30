import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# loading the data
data = pd.read_csv('dataset.csv')


#imputing missing values for the categorical varaibles
cat_list = ['location_region', 'location_state','customer_value','gender','device_type','device_manufacturer']
for a in cat_list:
    data[a].fillna('unspecified',inplace=True)


#imputing missing values for the numerical variables
num_list = ['spend_total', 'spend_vas', 'spend_voice', 'spend_data','xtra_data_talk_rev', 'customer_class','age']
for b in num_list:
    data[b].fillna(data[b].median(), inplace=True)


data['sms_cost'].fillna(value=0, inplace=True)
data['event_type'].fillna(data['event_type'].mode()[0], inplace=True)
        
        
#Reducing the categories in device_manufacturer column        
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
        

#Encoding the target variable
for k in data['event_type']:
    if k == 'Click':
        data['event_type'].replace(k,1,inplace=True)
    else:
        data['event_type'].replace(k,0,inplace=True)
        
# Encoding the customer_value variable        
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
        


# import packages for encoding of the categorical variables
# One Hot Encoding
X = pd.get_dummies(data, columns=cat_list, dummy_na=True)


#Dropping id's and empty variables
list = ['msisdn', 'location_lga','location_city', 'os_name','os_version',
       'ad_id', 'ad_name', '@timestamp', 'event_type']
y = X.loc[:,'event_type']
X = X.drop(list,axis = 1 )


# split in train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Model building using Random Forest algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,recall_score,precision_score

rf = RandomForestClassifier(random_state=43)      
rf = rf.fit(X_train,y_train)


cm = confusion_matrix(y_test,rf.predict(X_test))
sns.heatmap(cm,annot=True,fmt="d")

print("Precision: %.3f" % precision_score(y_test, rf.predict(X_test)))
print("Recall: %.3f" % recall_score(y_test, rf.predict(X_test)))



# saving the model
import pickle
with open('model/model.pkl','wb') as file:
    pickle.dump(rf, file)
    
    
    
# saving the columns
model_columns = X.columns
with open('model/model_columns.pkl','wb') as file:
    pickle.dump(model_columns, file)