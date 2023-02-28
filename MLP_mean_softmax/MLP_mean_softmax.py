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
import datetime
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter

from tensorflow.keras import initializers
from contextlib import redirect_stdout

#==============================================================================
#============================== FUNCTIONS =====================================
#==============================================================================
#ERROR DNN library is not found 4920
tf.config.experimental.set_visible_devices([], 'GPU')

print("\n====================\n")
print("The version is:")
print(tf.__version__)
EPOCH =300
BATCH = 5
X = 500
Y = 50
maxValueHist = 2500
minValueHist = -2500

FOLDER = "./tfg1/tfg1.1/"


def entropy(labels, base=2):
    _,counts = np.unique(labels, return_counts=True)
    return scipy.stats.entropy(counts, base=base)


def matrix_array_onelayer3(capa, picture, vector, pX, pY, final):
    imgaux = cp.deepcopy(picture[capa, pY:,pX:final])
    featureValuesN = []
    targetValuesN = []
    y = vector[0]
    x = vector[1]
    
    for j in range(imgaux.shape[0]):
        for i in range(imgaux.shape[1]):
            a = []
            targetValuesN.append(imgaux[j][i])
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
            
            
            featureValuesN.append(a)
    return np.array(featureValuesN),np.array(targetValuesN)

def mse(y, y_):
    aux = np.subtract(y, y_)
    mseV = np.power(aux, 2)
    return mseV

def mseAll (A,B):
    return np.mean(np.square(A - B))



#==============================================================================
#================================ IMAGES ======================================
#==============================================================================


training_images = glob.glob("*224x512x680.raw")
img = fileio.loadImage(training_images[0])[0].astype('uint16')
mean_value = np.mean(img)
img  = img - mean_value
img = np.around(img)
img = np.clip(img, minValueHist, maxValueHist)
img = np.array(img, dtype=np.int64)
#============================= INFORMATION ====================================
#                   ============= Softmax =============

minValue = np.min(img)
maxValue = maxValueHist
rangeV = 1 + maxValue - minValue

pX = 1
pY = 1
pZ = 0
pdXr = 4


vector = [0,-1]
vector = np.array(vector)
name = FOLDER + "logs"
tf.keras.backend.set_floatx('float64')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=name, histogram_freq=1, profile_batch = 0)
for BAND in range(5, 200):
    try:
        featuresValues, targetValues = matrix_array_onelayer3(BAND, img[:, :400, :], vector, 3, 3, X)

        #==============================================================================
        #=========================== NEURONAL NETWORK =================================
        #==============================================================================

        # Features Normalization
        featuresValues = np.clip(featuresValues, minValueHist, maxValueHist)
        featuresValues = featuresValues/2**12
        targetValues = np.reshape(targetValues, (targetValues.shape[0],1))
        targetValues2 = targetValues
        targetValues2 = tf.keras.utils.to_categorical(targetValues2, num_classes=rangeV, dtype='float64')
        featuresValues = tf.convert_to_tensor(featuresValues, dtype=tf.float64)
        
        #============================= INFORMATION ====================================

        print("====================================================================")
        print("===========================INFORMATION==============================")
        print("====================================================================")

        print("\n The shapes are: ")
        print("FeaturesValues: ")
        print(featuresValues.shape)
        print("targetValues2: ")
        print(targetValues2.shape)
        print(featuresValues[0])
        print(targetValues2[0])
        #=========================== NEURONAL NETWORK =================================

        # x = np.random.rand(100, 13)
        # y = np.array(np.full((100,1), 0.3), dtype=np.float64)
        # print(y)
        # featuresValues = np.concatenate((x, y), axis=1)
        # print(featuresValues[-1])
        # targetValues2 =  np.around(featuresValues.T[-1] * rangeV)
        # targetValues2 = targetValues2.T
        # targetValues2 = tf.keras.utils.to_categorical(targetValues2, num_classes=(rangeV), dtype='float64')
        
        # print(np.around(featuresValues.T[-1]*rangeV))
        print("BAND")
        print(BAND)
        print("============================FIRSTTIME==============================")
        model = tf.keras.Sequential()
        model.add(keras.Input(shape=(14)))
        model.add(layers.Dense(units = 8000, activation = tf.nn.relu, use_bias=True))
        model.add(layers.Dense(units = 5001, activation = tf.nn.relu, use_bias=True))
        model.add(layers.Dense(units = rangeV,  activation = tf.nn.softmax, use_bias=True))
        model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        historial = model.fit(featuresValues, targetValues2, epochs=30, batch_size=5000, callbacks=[tensorboard_callback])
        model.summary()
        score = model.evaluate(featuresValues, targetValues2, verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        print("====================================================================")
        print("============================PREDICTING==============================")
        print("====================================================================")

        print("=========================================")
        print("=========================================")
        print("=========================================")
        print("=========================================")
        print("BAND")
        print(BAND)
        print("=========================================")
        print("=========================================")
        print("=========================================")
        print("=========================================")

        featuresValues, targetValues = matrix_array_onelayer3(BAND, img[:, :400, :], vector, 3, 3, X)
        # Features Normalization
        featuresValues = np.clip(featuresValues, minValueHist, maxValueHist)
        featuresValues = featuresValues/2**12
        targetValues = np.reshape(targetValues, (targetValues.shape[0],1))
        featuresValues = tf.convert_to_tensor(featuresValues, dtype=tf.float64)


        ArraypredictValue = []
        count = 0
        for i in featuresValues:
                i = tf.expand_dims(i, axis=0)
                softmaxValue = model.call(i)
                ArraypredictValue.append(tf.math.argmax(softmaxValue[0]))
        
        print("============================ENTROPY===============================")
        print(ArraypredictValue)
        ArraypredictValue = np.array(ArraypredictValue)
        ArraypredictValue = np.around(ArraypredictValue)
        allResult = []
        for i in range(len(ArraypredictValue)):
             allResult.append(ArraypredictValue[i]-targetValues[i])
        mseValue = np.around(allResult)
        entropyValue = entropy(allResult)
        MSE = np.average(allResult)

        print("")
        print(entropyValue)
        print(MSE)
        name = 'MLP_Softmax/'+str(BAND)+'.txt'
        file = open(name, "w")
        file.write("Entropy test:")
        file.write(str(entropyValue))
        file.write("\n")
        file.write("MSE test:")
        file.write(str(MSE))
        file.write("\n")
        file.close()
    except Exception as e:
        print(e)
        print("ERROR")
