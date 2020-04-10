# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:25:38 2020

@author: NARAYANA REDDY DATA SCIENTIST
"""


# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# READ THE DATA SET
dataset=pd.read_csv('Social_Network_Ads.csv')

# DIVIDE THE DATA SET INTO X AND Y
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

# SPLITTING THE DATA SET INTO TRAIN AND TEST DATASET
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# FFEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

# FITTING THE RANDOM FOREST MODEL TO TRAINING DATASET
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

y_predict=classifier.predict(x_test)
# PERFORMANCE METRICS FOR CLASSIFICATION MODEL
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_predict)
accuracyscore=accuracy_score(y_test,y_predict)

# RANDOMIZEDSEARCHCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


est=RandomForestClassifier(n_jobs=-1)
rf_p_dist={'max_depth':[3,5,9,None],
           'n_estimators':[100,200,300,400,500],
           'max_features':randint(1,3),
           'criterion':['gini','entropy'],
           'bootstrap':[True,False],
           'min_samples_leaf':randint(1,4)
           }

def hypertuning_rscv(est,p_distr,nbr_iter,x,y):
    rdmsearch = RandomizedSearchCV(est,param_distributions=p_distr,
                                 n_jobs=-1,n_iter=nbr_iter,cv=10)
    
    rdmsearch.fit(x,y)
    ht_params=rdmsearch.best_params_
    ht_score=rdmsearch.best_score_
    return ht_params,ht_score

rf_parameters,rf_ht_score=hypertuning_rscv(est,rf_p_dist,40,x,y)

# TUNE THE RANDOMFOREST ALGORITH USING HYPERPARAMETERS

calssifier=RandomForestClassifier(n_estimators=400,
                                  bootstrap=True,
                                  criterion='entropy',
                                  max_depth=3,max_features=2,
                                  min_samples_leaf=1,
                                  random_state=0)

classifier.fit(x_train,y_train)

y_predict=classifier.predict(x_test)

# PERFORMANCE METRICS FOR CLASSIFICATION MODEL
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_predict)
accuracyscore=accuracy_score(y_test,y_predict)
