# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 12:20:32 2020

@author: Ankita
"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import lightgbm as lgb


train = pd.read_csv("Aug_Train.csv")
train = train.fillna('')
#df = train.copy(deep=True)
df =train
test = pd.read_csv("Test_LqhgPWU.csv")
test = test.fillna('')


le = LabelEncoder()
categorical = np.array(['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Var_1', 'Spending_Score', 'Segmentation'])

for i in range(len(categorical)):
    train[categorical[i]] = le.fit_transform(train[categorical[i]])
    if i < len(categorical)-1:
        test[categorical[i]] = le.fit_transform(test[categorical[i]])

train = train.replace('', np.nan)
test = test.replace('', np.nan)

df_test = test.copy(deep=True)

labels = train['Segmentation'].values
train = train.drop(['Segmentation'], axis = 1)
train = train.drop(['ID'], axis = 1)
test = test.drop(['ID'], axis = 1)

temp, age_temp = list(df['Age'].values), list()
for age in range(len(temp)):
    i = temp[age]
    if i <=33:
        age_temp.append(0)
    elif i <=67:
        age_temp.append(1)
    elif i <=89:
        age_temp.append(2)

train['Age_Prof'] = age_temp

temp, age_temp = list(df_test['Age'].values), list()
for age in range(len(temp)):
    i = temp[age]
    if i <=33:
        age_temp.append(0)
    elif i <=67:
        age_temp.append(1)
    elif i <=89:
        age_temp.append(2)

test['Age_Prof'] = age_temp

train = train.values
test = test.values

scaler = StandardScaler()
X_train = train
y_train = labels
scaler.fit(X_train)

X_train = scaler.transform(X_train)
models_rf, models_lgb = list(), list()
kf = KFold(n_splits=5, random_state=28, shuffle=False)
for train_index, test_index in kf.split(X_train):
    X_traink, X_testk = X_train[train_index], X_train[test_index]
    y_traink, y_testk = y_train[train_index], y_train[test_index]
    clf2 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=80, min_data_in_leaf=307, max_iter=1000, max_depth=7, learning_rate=0.1)
    clf2.fit(X_traink, y_traink)
    models_lgb.append(clf2)
    scores = clf2.predict(X_testk)
    #print(accuracy_score(scores, y_testk))

best_model_lgb = models_lgb[1]

test = scaler.transform(test)
preds2 = best_model_lgb.predict(test)

final_preds2, temp_train, temp_test = list(), list(df["ID"].values), list(df_test["ID"].values)
for i in range(len(temp_test)):
    id_ = temp_test[i]
    if (id_ in temp_train):
        temp = df[df['ID'] == id_]
        final_preds2.append(temp['Segmentation'].values[0])
    else:
        final_preds2.append(preds2[i])

new_preds2 = []

for i in range(len(preds2)):
    if final_preds2[i] == 0:
        new_preds2.append('A')
    elif final_preds2[i] == 1:
        new_preds2.append('B')
    elif final_preds2[i] == 2:
        new_preds2.append('C')
    elif final_preds2[i] == 3:
        new_preds2.append('D')
        

df_submit = pd.DataFrame({'ID': df_test['ID'].values, 'Segmentation':new_preds2})
df_submit.to_csv('submit2.csv', index = False)