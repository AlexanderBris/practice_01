# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:21:01 2023

@author: Alexander
"""

# подгрузка стандартных библиотек
import os
import numpy as np
import pandas as pd

# функция чтения данных из файла
def read_data(path, filename):
    return pd.read_csv(os.path.join(path, filename))

# загружаем данные в переменную df
cur_dir = os.getcwd()
# файлы лежат в папке с основным скриптом
df = read_data(cur_dir, 'train.csv')
# проверка
df.head()

def load_dataset(label_dict):
    train_X = read_data(cur_dir, 'train.csv').values[:,:-2]
    train_y = read_data(cur_dir, 'train.csv')['Activity']
    train_y = train_y.map(label_dict).values
    test_X = read_data(cur_dir, 'test.csv').values[:,:-2]
    test_y = read_data(cur_dir, 'test.csv')
    test_y = test_y['Activity'].map(label_dict).values
    return(train_X, train_y, test_X, test_y)    

label_dict = {'WALKING':0, 'WALKING_UPSTAIRS':1, 'WALKING_DOWNSTAIRS':2, 'SITTING':3, 'STANDING':4, 'LAYING':5}

# создаем списки для данных
train_X, train_y, test_X, test_y = load_dataset(label_dict)

# берем модели из sklearn

#------------------------------------------------------------------
#------------------------------------------------------------------
from sklearn.neighbors import  KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7, weights='uniform', 
                             algorithm='auto', leaf_size=20,
                             p=2, metric='nan_euclidean', metric_params=None,
                             n_jobs=None)

# обучаем алгоритмы
model.fit(train_X, train_y)     
ywhat = model.predict(test_X)

from sklearn.metrics import classification_report
target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']
print("KNeighborsClassifier")
print(classification_report(test_y, ywhat, target_names=target_names))    

    
#------------------------------------------------------------------
#------------------------------------------------------------------
from sklearn.neighbors import  RadiusNeighborsClassifier
model = RadiusNeighborsClassifier(radius=500.0, weights='uniform',
                                  algorithm='brute', leaf_size=30,
                                  p=2, metric='minkowski',
                                  outlier_label=None,
                                  metric_params=None, n_jobs=None)

# обучаем алгоритмы
model.fit(train_X, train_y)     
ywhat = model.predict(test_X)

from sklearn.metrics import classification_report
target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']
print("RadiusNeighborsClassifier")
print(classification_report(test_y, ywhat, target_names=target_names))    


#------------------------------------------------------------------
#------------------------------------------------------------------
from sklearn.tree import  ExtraTreeClassifier
model = ExtraTreeClassifier(criterion='gini',
                                  splitter='random', max_depth=None,
                                  min_samples_split=2, min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0,
                                  max_features='sqrt', random_state=None,
                                  max_leaf_nodes=None,
                                  min_impurity_decrease=0.0,
                                  class_weight=None, ccp_alpha=0.0)

# обучаем алгоритмы
model.fit(train_X, train_y)     
ywhat = model.predict(test_X)

from sklearn.metrics import classification_report
target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']
print("ExtraTreeClassifier")
print(classification_report(test_y, ywhat, target_names=target_names))    


#------------------------------------------------------------------
#------------------------------------------------------------------
from sklearn.neural_network import  MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(100,),
                      activation='relu', solver='adam',
                      alpha=0.0001, batch_size='auto',
                      learning_rate='constant', learning_rate_init=0.001,
                      power_t=0.5, max_iter=200, shuffle=True,
                      random_state=None, tol=0.0001, verbose=False,
                      warm_start=False, momentum=0.9,
                      nesterovs_momentum=True, early_stopping=False,
                      validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                      epsilon=1e-08, n_iter_no_change=10, max_fun=15000)


# обучаем алгоритмы
model.fit(train_X, train_y)     
ywhat = model.predict(test_X)

from sklearn.metrics import classification_report
target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']
print("MLPClassifier")
print(classification_report(test_y, ywhat, target_names=target_names))    
