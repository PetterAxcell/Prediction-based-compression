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
X = 300
Y = 50
maxValueHist = 10000

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
training_images[0]
img = fileio.loadImage(training_images[0])[0].astype('uint16')
img.shape

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
    # try:
        featuresValues, targetValues = matrix_array_onelayer3(BAND, img[:, :512, :680], vector, 3, 3, X)

        #==============================================================================
        #=========================== NEURONAL NETWORK =================================
        #==============================================================================

        # Features Normalization
        featuresValues = np.clip(featuresValues, 0, maxValueHist)
        featuresValues = featuresValues/2**14
        targetValues = np.reshape(targetValues, (targetValues.shape[0],1))
        targetValues2 = targetValues-minValue
        targetValues2 = np.clip(targetValues2, 0, maxValueHist)
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
        model.add(layers.Dense(units = 50, kernel_initializer = tf.keras.initializers.Zeros, activation = tf.nn.relu, use_bias=True))
        model.add(layers.Dense(units = 50, kernel_initializer = tf.keras.initializers.Zeros,  activation = tf.nn.relu, use_bias=True))
        model.add(layers.Dense(units = rangeV,  activation = tf.nn.softmax, use_bias=True))
        model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        historial = model.fit(featuresValues, targetValues2, epochs=3, batch_size=196, callbacks=[tensorboard_callback])
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

        ArraypredictValue = []
        count = 0
        for i in featuresValues:
                i = tf.expand_dims(i, axis=0)
                softmaxValue = model.call(i)
                print("The softmaxCalculated:")
                print(tf.math.argmax(softmaxValue[0]))
                print("Max value")
                print(tf.math.reduce_max(softmaxValue[0]))
                ArraypredictValue.append(tf.math.argmax(softmaxValue[0]))
        
        print("============================ENTROPY===============================")
        print(ArraypredictValue)
        break
    #     ArraypredictValue = np.array(ArraypredictValue)
    #     ArraypredictValue = np.around(ArraypredictValue)
    #     print(ArraypredictValue)
    #     print(ArraypredictValue.shape)
    #     MSEIndividual = mse(targetValues,ArraypredictValue)
    #     mseValue = np.around(MSEIndividual)
    #     entropyValue = entropy(MSEIndividual)
    #     MSE = np.average(MSEIndividual)

    #     print("")
    #     print(entropyValue)
    #     print(MSE)
    #     name = 'MLP/'+str(BAND)+'.txt'
    #     file = open(name, "w")
    #     file.write("Entropy test:")
    #     file.write(str(entropyValue))
    #     file.write("\n")
    #     file.write("MSE test:")
    #     file.write(str(MSE))
    #     file.write("\n")
    #     file.close()
    # except Exception as e:
    #     print(e)
    #     print("ERROR")
