import os
import tensorflow as tf
import fileio
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
import itertools
import copy as cp
import random
import datetime
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras import initializers
from contextlib import redirect_stdout



# ------------------ FUNCTIONS ------------------ 
def entropy(labels, base=2):
    _,counts = np.unique(labels, return_counts=True)
    return scipy.stats.entropy(counts, base=base)


def mse(y, y_):
    aux = y-y_
    mseV = np.power(aux, 2)
    return mseV

def mseAll (A,B):
    return np.mean(np.square(A - B))

def matrix_array_onelayer(picture, vector, pX, pY):
    imgaux = cp.deepcopy(picture[pY:,pX:])
    featuresValues = []
    targetValues = []
    y = vector[0]
    x = vector[1]
    for j in range(imgaux.shape[0]):
        for i in range(imgaux.shape[1]):
            targetValues.append(imgaux[j][i])
            featuresValues.append(picture[j][i])
    return np.array(featuresValues),np.array(targetValues)

def matrix_array_onelayer2(picture, vector, pX, pY):
    imgaux = cp.deepcopy(picture[pY:,pX:])
    featuresValues = []
    targetValues = []
    y = vector[0]
    x = vector[1]
    
    for j in range(imgaux.shape[0]):
        for i in range(imgaux.shape[1]):
            a = []
            targetValues.append(imgaux[j][i])
            a.append(picture[j][i])
            a.append(picture[j][i+1])
            featuresValues.append(a)
    return np.array(featuresValues),np.array(targetValues)

def matrix_array_onelayer3(capa, picture, vector, pX, pY, final):
    imgaux = cp.deepcopy(picture[capa, pY:,pX:final])
    featuresValues = []
    targetValues = []
    y = vector[0]
    x = vector[1]
    
    for j in range(imgaux.shape[0]):
        for i in range(imgaux.shape[1]):
            a = []
            targetValues.append(imgaux[j][i])
            #Misma fila
            a.append(picture[capa][j+3][i+1])
            a.append(picture[capa][j+3][i+2])
            #Misma fila
            a.append(picture[capa][j+2][i+1])
            a.append(picture[capa][j+2][i+2])
            a.append(picture[capa][j+2][i+3])
            a.append(picture[capa][j+2][i+4])
            #Misma fila
            a.append(picture[capa][j+1][i+2])
            a.append(picture[capa][j+1][i+3])
            a.append(picture[capa][j+1][i+4])
            
            
            #Anterior capa
                #Misma fila
            a.append(picture[capa-1][j+3][i+2])
            a.append(picture[capa-1][j+3][i+3])
                 #Misma fila
            a.append(picture[capa-1][j+2][i+2])
            a.append(picture[capa-1][j+2][i+3])
            a.append(picture[capa-1][j+2][i+4])
            
            #Anterior capa
                #Misma fila
            #a.append(picture[capa-2][j+3][i+2])
            #a.append(picture[capa-2][j+3][i+3])
                 #Misma fila
            #a.append(picture[capa-2][j+2][i+2])
            #a.append(picture[capa-2][j+2][i+3])
            #a.append(picture[capa-2][j+2][i+4])
            
            
            featuresValues.append(a)
    return np.array(featuresValues),np.array(targetValues)



def matrix_array(picture, vector, padX, padY, padZ, padXr):
    numXr = picture.shape[2]-padXr
    new_picture = np.array(picture[padZ:, padY:, padX:numXr])
    feature_value = []
    target_value = []
    for k in range(new_picture.shape[0]):
        for j in range(new_picture.shape[1]):
            for i in range(new_picture.shape[2]):
                a = []
                for iVec in vector:
                    z = k + padZ
                    y = j + padY
                    x = i + padX-1
                    a.append(picture[z+iVec[0],y+iVec[1],x+iVec[2]])
                feature_value.append(a)
                target_value.append(new_picture[k, j, i])
        print("Scope number: ", k+padZ)
    return np.array(feature_value), np.array(target_value)

# ------------------ IMAGES ------------------ 

training_images = glob.glob("*224x512x680.raw")
training_images[0]
img = fileio.loadImage(training_images[0])[0].astype('uint16')

vector = [0,-1]
vector = np.array(vector)


pX = 1
pY = 1
pZ = 0
pdXr = 4

# ------------------ LeastSquare Trainning ------------------ 

Entropies=[]
imatgeReal = []
Mse = []
for i in range(70):
    i = i+5
    featuresValues, targetValues = matrix_array_onelayer3(i, img[:, :512, :680], vector, 3, 3, 678)
    A = np.insert(featuresValues,featuresValues.shape[1], 1, axis=1)
    result = np.linalg.lstsq(A, targetValues, rcond=None)
    a = []
    for j in range(len(result[0])):
        a.append(result[0][j])

    w = np.array(a)
    w = w.reshape(1,15)
    
    predictValue = np.around(w*A)
    predictValue= predictValue.sum(axis=1)
    imatgeReal.append(predictValue.reshape(509,675))
    mseValue = np.around(mse(targetValues, predictValue))
    Entropies.append(entropy(mseValue))
    Mse.append(np.around(mseAll(targetValues, predictValue)))
    print(i)
    

for i in range(70):
    i = i+75
    featuresValues, targetValues = matrix_array_onelayer3(i, img[:, :512, :680], vector, 3, 3, 678)
    A = np.insert(featuresValues,featuresValues.shape[1], 1, axis=1)
    result = np.linalg.lstsq(A, targetValues, rcond=None)
    a = []
    for j in range(len(result[0])):
        a.append(result[0][j])

    w = np.array(a)
    w = w.reshape(1,15)
    
    predictValue = np.around(w*A)
    predictValue= predictValue.sum(axis=1)
    imatgeReal.append(predictValue.reshape(509,675))
    mseValue = np.around(mse(targetValues, predictValue))
    Entropies.append(entropy(mseValue))
    Mse.append(np.around(mseAll(targetValues, predictValue)))
    print(i)

for i in range(70):
    i = i+145
    featuresValues, targetValues = matrix_array_onelayer3(i, img[:, :512, :680], vector, 3, 3, 678)
    A = np.insert(featuresValues,featuresValues.shape[1], 1, axis=1)
    result = np.linalg.lstsq(A, targetValues, rcond=None)
    a = []
    for j in range(len(result[0])):
        a.append(result[0][j])

    w = np.array(a)
    w = w.reshape(1,15)
    
    predictValue = np.around(w*A)
    predictValue= predictValue.sum(axis=1)
    imatgeReal.append(predictValue.reshape(509,675))
    mseValue = np.around(mse(targetValues, predictValue))
    Entropies.append(entropy(mseValue))
    Mse.append(np.around(mseAll(targetValues, predictValue)))
    print(i)


print(np.average(Entropies))
print(np.average(Mse))

Predict = "".join(['\n' + str(_) for _ in Entropies])
Predict2 = "".join(['\n' + str(_) for _ in Mse])
# Write
file = open('LeastSquare.txt', "w")
file.write("Entropy test:")
file.write(Predict)
file.write("\n MSE test:")
file.write(Predict2)
file.close()