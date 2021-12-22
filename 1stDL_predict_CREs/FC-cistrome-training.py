#!/usr/bin/env python
# coding: utf-8

from optparse import OptionParser
usage = "USAGE: this.py [-p] [-n] [-o] [-e] [-l]"
parser = OptionParser(usage=usage)
parser.add_option("-p", dest="posi", action="store",help="File path to positive tiles")
parser.add_option("-n", dest="nega", action="store", help="File path to negative tiles")
parser.add_option("-o", dest="out", action="store", help="output prefix")
parser.add_option("-e", dest="epoch", action="store", help="epoch numbers")
parser.add_option("-l", dest="length", action="store", help="DNA length")

(opt, args) = parser.parse_args()

posi = opt.posi
nega = opt.nega
out = opt.out
epoch = opt.epoch
length = opt.length


import numpy as np
import os
import re
import numpy as np

import tensorflow as tf
import keras
from keras.layers import (Activation, Add, GlobalAveragePooling2D,
                          BatchNormalization, Conv1D, Conv2D, Dense, Flatten, Reshape, Input, Dropout,
                          MaxPooling1D,MaxPooling2D)
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.utils import plot_model,np_utils
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from keras.models import Model, load_model
from tqdm import tqdm
from functools import reduce

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix


input_shape=(int(length),4,1)
num_classes=2
epochnum = int(epoch)

trained_model_filename = out + '.h5'
trained_model_filepath = './models/'+trained_model_filename
trained_models_weight_dir ='./models/tensorlog/' + out + 'weights.hdf5'

def visualize_loss_acc(history):
    import matplotlib.pyplot as plt
    
    # Setting Parameters
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # 1) Accracy Plt
    plt.plot(epochs, acc, 'bo' ,label = 'training acc')
    plt.plot(epochs, val_acc, 'ro' , label= 'validation acc')
    plt.title('Train/Validation_loss/acc')
    plt.ylim(0,1.1)
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.show()
    
    
    plt.plot(epochs, loss, 'b' ,label = 'training loss')
    plt.plot(epochs, val_loss, 'r' , label= 'validation loss')    
    plt.title('Train/Validation_loss/acc')
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.show()


def dna2num(dna):
    if dna.upper() == "A":
        return 0
    elif dna.upper() == "T":
        return 1
    elif dna.upper() == "G":
        return 2
    else:
        return 3

def num2dna(num):
    if num == 0:
        return "A"
    elif num == 1:
        return "T"
    elif num == 2:
        return "G"
    else:
        return "C"
    
def dna2array(DNAstring):
    numarr = []
    length = len(DNAstring)
    for i in range(0, length):
        num = dna2num(DNAstring[i:i+1])
        if num >= 0:
            numarr.append(num)
    return numarr

def array2dna(numarr):
    DNAstring = []
    length = numarr.shape[0]
    for i in range(0, length):
        dna = num2dna(numarr[i].argmax())
        DNAstring.append(dna)
    DNAstring = ''.join(DNAstring)
    return DNAstring


def load_data():
    X = []
    Y = []
    f = open(posi, "r")
    line=f.readline()
    while line:
        line2 = line.rstrip()
        OneHotArr = np.array([np.eye(4)[dna2array(line2)]])
        X.extend(OneHotArr)
        Y.append(1)
        line = f.readline()
    f = open(nega, "r")
    line=f.readline()
    while line:
        line2 = line.rstrip()
        OneHotArr = np.array([np.eye(4)[dna2array(line2)]])
        X.extend(OneHotArr)
        Y.append(0)
        line = f.readline()
    X = np.array(X)
    Y = np.array(Y)
    Y = np_utils.to_categorical(Y, 2)
    
    X = np.reshape(X,(-1, int(length), 4, 1))
    
    return (X, Y)

def get_FC_3layer(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape[1:]))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    return model

"""===Load_Dna_Sequences==="""
X, Y = load_data()
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=111)


"""===Calculate_ClassWeight==="""
y_integers = np.argmax(Y, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))


"""===Compile_Models==="""
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999,epsilon=None,decay=0.0)

model = get_FC_3layer(input_shape = X_train.shape)
trained_models_dir = './models/'
dirpath = trained_models_dir + 'tesorlog/'

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

class_weight = d_class_weights,
print(class_weight)

"""===Start_Train==="""
history = model.fit(
    X_train,
    Y_train,
    batch_size=32,
    initial_epoch=0,
    epochs=epochnum,
    validation_data=(X_test, Y_test),
    class_weight = d_class_weights,
    )

"""===Save_Weights==="""
model.save(trained_model_filepath)

loss, acc = model.evaluate(X_test, Y_test)

"""======Load Model======"""
print("load_model")
trained_models_dir = './models/tensorlog/'
dirpath = trained_models_dir + 'fullyconnected/'

"""===Load_Dna_Sequences==="""
X, Y = load_data()
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=111)

"""===Predict Data"""
cm_roc = open(out + "_CM-ROC.txt", "w")
prediction = model.predict(X_test,verbose=0)
y_pred = np.argmax(prediction, axis=1)
y_val = np.argmax(Y_test, axis=1)
y_pred_value = [prediction[i][1] for i in range(y_pred.shape[0])]

""""======confusion matrix and Roc AUC======="""
print(confusion_matrix(y_val, y_pred), file=cm_roc)
roc_val = roc_auc_score(y_val, y_pred_value)

cm_roc.write(out + '_roc-auc_val: %s' % (str(round(roc_val,4))))
cm_roc.close()



