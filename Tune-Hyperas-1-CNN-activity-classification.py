#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/khataei/PE-classification-DeepLearning/blob/master/Tunned-Talos-1-CNN-activity-classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Hyperas Tuner for CNN Activity Classifier

# In this notebook, we use [Hyperas](https://github.com/maxpumperla/hyperas) to tune a CNN neural net to classify PE activity.

# #### Load dependencies

# In[18]:


from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, GlobalMaxPooling1D, MaxPool1D, GlobalAveragePooling1D
from tensorflow.keras.layers import AveragePooling1D, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint 
import os  
from sklearn.metrics import roc_auc_score, roc_curve 
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# tunning imports
  
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.models import Sequential
from hyperas.distributions import randint


# #### Set hyperparameters

# In[10]:


# output directory name:
output_dir = 'model_output/cnn2'
input_dir =  'Z:/Research/dfuller/Walkabilly/studies/smarphone_accel/data/Ethica_Jaeger_Merged/pocket/'
input_file_name = 'pocket-NN-data.npz'

# from the data preparation section we have:
window_size_second = 3
frequency = 30
lenght_of_each_seq = window_size_second * frequency


# ##### For this notebook we use the acceleration data gathered from the pocket location. It was prepared in the DataPrep-Deep notebook

# #### Load data

# In the raw data format of the labels is String and there are 6 classes. 'Lying', 'Sitting', 'Self Pace walk', 'Running 3 METs',
#        'Running 5 METs', 'Running 7 METs' <br>
# 
# 
# 

# In[22]: 


def data():
    output_dir = 'model_output/cnn2'
    input_dir =  'Z:/Research/dfuller/Walkabilly/studies/smarphone_accel/data/Ethica_Jaeger_Merged/pocket/'
    input_file_name = 'pocket-NN-data.npz'
    # read the raw file and get the keys:
    raw_data = np.load(file=input_dir+input_file_name,allow_pickle=True)
    for k in raw_data.keys():
        print(k)
    accel_array = raw_data['acceleration_data']
    meta_array = raw_data['metadata']
    labels_array = raw_data['labels']

    
    # change from string to integer so keras.to_categorical can consume it

    # could do with factorize method as well
    n_class = len(np.unique(labels_array))
    class_list, labels_array_int = np.unique(labels_array,return_inverse=True)


    y = to_categorical(labels_array_int, num_classes=n_class)

    
    # split and shuffle
    X_train, X_valid, y_train, y_valid = train_test_split(
     accel_array, y, test_size=0.1, random_state=65)
    
    return X_train, y_train, X_valid, y_valid


# 
# #### Design neural network architecture

# In[23]:


#hyperas hyper params

# choice()
# pooling layer parameters
maxpooling_pool_size = 2
avepooling_pool_size = 2


# convolutional layer architecture:
#n_conv_1 = choice([256, 512, 1024]) # filters, a.k.a. kernels
#n_conv_1 = choice([256, 512, 1024]) # filters, a.k.a. kernels
#k_conv_1 = choice([2,10]) # kernel length
# n_conv_2 = choice('k_conv_2', [256, 512, 1024]) # filters, a.k.a. kernels
# k_conv_2 = choice('k_conv_2', [2, 4, 8]) # kernel length
# n_conv_3 = choice('k_conv_3', [256, 512, 1024]) # filters, a.k.a. kernels
# k_conv_3 = choice('k_conv_3', [2, 4, 8]) # kernel length

n_conv_2 = 256
k_conv_2 = 3 # kernel length
n_conv_3 = 256 # filters, a.k.a. kernels
k_conv_3 = 2 # kernel length


#dense layer architecture: 
n_dense_1 = 512
dropout_1 = 0.3
n_dense_2 = 256
dropout_2 = 0.25

# training:
epochs = 60
batch_size = 256


# In[24]:


def model(X_train, y_train, X_valid, y_valid):
    '''
    Model providing function:
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    model = Sequential()
    input_shape = list(X_train.shape)

    model.add(tf.keras.layers.Conv1D({{choice([256, 512, 1024])}}, {{choice([2,4])}}, activation='relu', input_shape=input_shape[1:]))
    model.add(tf.keras.layers.MaxPool1D(pool_size = maxpooling_pool_size))
    model.add(tf.keras.layers.Conv1D(n_conv_2, k_conv_2, activation='relu'))
    model.add(tf.keras.layers.AveragePooling1D(pool_size = avepooling_pool_size))
    model.add(tf.keras.layers.Conv1D(n_conv_3, k_conv_3, activation='relu'))
    # model.add(GlobalMaxPooling1D())
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(n_dense_1, activation=LeakyReLU(alpha=0.1)))
    model.add(tf.keras.layers.Dropout(dropout_1))
    model.add(tf.keras.layers.Dense(n_dense_2, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_2))
    model.add(tf.keras.layers.Dense(n_class, activation='softmax'))
    print(model.summary())

    
    
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer={{choice(['rmsprop', 'adam'])}},
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size={{choice([128, 256])}},
              nb_epoch=1,
              verbose=2,
              validation_data=(X_valid, y_valid))
    score, acc = model.evaluate(X_valid, y_valid, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


# In[26]:



if __name__ == '__main__':

    trials = Trials()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=trials)
    for trial in trials:
        print(trial)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))

