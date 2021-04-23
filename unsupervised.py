
import os
# Force Keras on CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2    
import matplotlib.pyplot as plt   
import pandas as pd
from keras.preprocessing import image   
import numpy as np   
from keras.utils import np_utils
from skimage.transform import resize  
from openpyxl import load_workbook

from keras.layers import TimeDistributed, GRU, Dense, Dropout, LayerNormalization, ConvLSTM2D

from keras.layers import Conv2D, BatchNormalization, \
    MaxPool2D, GlobalMaxPool2D

import keras

import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose

# Fichier pour faire des tests d'apprentissages supervisés, seulement sur les images pour l'instant


print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices())

# nombre de frames par séquence
NBFRAME = 5
# taille des images après redimensionnement pour fit
SIZE = (112, 112)
# TODO : convertir image en greyscale pour entrainement


# création des séquences d'images pour entrainement et validation

def defineTrainSets(NBFRAME = NBFRAME, SIZE = SIZE):
    trainList = []
    testList = []

    anomaly_path = 'anomaly/'
    normal_path = 'normal/'
    count=0
    # for entry in os.listdir(anomaly_path):
    #     path = os.path.join(anomaly_path, entry)
    #     if os.path.isdir(path):
    #         if count == 4:
    #             testList.append(path)
    #         else :
    #             trainList.append(path)
    #         count+=1
    # count=0
    for entry in os.listdir(normal_path):
        path = os.path.join(normal_path, entry)
        if os.path.isdir(path):
            if count == 4:
                testList.append(path)
            else :
                trainList.append(path)
            count+=1

    trainListFrames = []
    trainListFramesClass = []
    for folder in trainList:
        className = folder.split('/')[0]
        print(className)
        imgs = os.listdir(folder)
        nbImgs = len(imgs)
        for i in range(nbImgs-NBFRAME):
            train_frames = []
            for j in range(NBFRAME):
                imgPath = folder+'/'+imgs[i+j]
                # loading the image and keeping the target size as (224,224,3)
                img = image.load_img(imgPath, target_size=SIZE+(3,))
                # img.show()
                # converting it to array
                img = image.img_to_array(img)
                # normalizing the pixel value
                img = img/255
                # appending the image to the train_image list
                train_frames.append(img)
            trainListFrames.append(train_frames)
            if className == 'anomaly':
                trainListFramesClass.append([0,1])
            else:
                trainListFramesClass.append([1,0])
    trainListFrames = np.array(trainListFrames)
    trainListFramesClass = np.array(trainListFramesClass)
    print(trainListFrames.shape)
    testListFrames = []
    testListFramesClass = []
    # for folder in testList:
    #     className = folder.split('/')[0]
    #     print(className)
    #     imgs = os.listdir(folder)
    #     nbImgs = len(imgs)
    #     for i in range(nbImgs-NBFRAME):
    #         train_frames = []
    #         for j in range(NBFRAME):
    #             imgPath = folder+'/'+imgs[i+j]
    #             # loading the image and keeping the target size as (224,224,3)
    #             img = image.load_img(imgPath, target_size=(112,112,3))
    #             # img.show()
    #             # converting it to array
    #             img = image.img_to_array(img)
    #             # normalizing the pixel value
    #             img = img/255
    #             # appending the image to the train_image list
    #             train_frames.append(img)
    #         testListFrames.append(train_frames)
    #         if className == 'anomaly':
    #             testListFramesClass.append([0,1])
    #         else:
    #             testListFramesClass.append([1,0])
    
    # testListFrames = np.array(testListFrames)
    # testListFramesClass = np.array(testListFramesClass)
    # print(testListFrames.shape)
    # print(testListFramesClass.shape)

    # y_train = np.asarray(trainListFramesClass).astype('float32').reshape((-1,1))
    # y_test = np.asarray(testListFramesClass).astype('float32').reshape((-1,1))
    # print(y_train.shape)
    # print(y_test.shape)

    return trainListFrames, testListFrames, trainListFramesClass, testListFramesClass

# création des séquences de données IMU pour entrainement et validation (pas fini!)
def defineTrainTestSetsIMU(NBFRAME = NBFRAME, SIZE = SIZE):
    trainList = []
    testList = []

    anomaly_path = 'anomalyIMU/'
    normal_path = 'normalIMU/'
    count=0
    for entry in os.listdir(anomaly_path):
        path = os.path.join(anomaly_path, entry)
        if os.path.isdir(path):
            if count == 4:
                testList.append(path)
            else :
                trainList.append(path)
            count+=1
    count=0
    for entry in os.listdir(normal_path):
        path = os.path.join(normal_path, entry)
        if os.path.isdir(path):
            if count == 4:
                testList.append(path)
            else :
                trainList.append(path)
            count+=1

    trainListSeries = []
    trainListSeriesClass = []
    for folder in trainList:
        className = folder.split('/')[0][:-3]
        print(className)
        data = os.listdir(folder)
        wb = load_workbook(filename = folder+'/' + data[0])
        sheets = wb.worksheets[1:]
        features = []
        for sheet in sheets:
            feature = []
            for val in sheet['B'][:15]:
                feature.append(val.value)
            features.append(feature)
        if className == 'anomaly':
            trainListSeriesClass.append([0,1])
        else:
            trainListSeriesClass.append([1,0])
        trainListSeries.append(features)
    

    trainListSeries = np.asarray(trainListSeries)
    trainListSeriesClass = np.array(trainListSeriesClass)

    testListSeries = []
    testListSeriesClass = []
    for folder in testList:
        className = folder.split('/')[0][:-3]
        print(className)
        data = os.listdir(folder)
        wb = load_workbook(filename = folder+'/' + data[0])
        sheets = wb.worksheets[1:]
        features = []
        for sheet in sheets:
            feature = []
            for val in sheet['B'][:15]:
                feature.append(val.value)
            features.append(feature)
        if className == 'anomaly':
            testListSeriesClass.append([0,1])
        else:
            testListSeriesClass.append([1,0])
    
    testListSeries = np.array(testListSeries)
    testListSeriesClass = np.array(testListSeriesClass)


    return trainListSeries, testListSeries, trainListSeriesClass, testListSeriesClass

# build_convnet_encoder et build_convnet_decoder ne sont pas utilisées pour l'instant,
# utilisation d'un autre autoencoder à la place
def build_convnet_encoder(shape=(112, 112, 3)):
    momentum = .9
    model = keras.Sequential()
    model.add(Conv2D(128, (3,3), input_shape=shape,
        padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    # flatten...
    model.add(GlobalMaxPool2D())
    return model

def build_convnet_decoder():
    momentum = .9
    model = keras.Sequential()
    model.add(Conv2DTranspose(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    model.add(Conv2DTranspose(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    # flatten...
    # model.add(GlobalMaxPool2D())
    return model

# autoencoder TimeDistributed Conv2D et ConvLSTM2D, 
def autoencoder_model(shape=(5, 112, 112, 3)):
    

    seq = keras.Sequential()
    seq.add(TimeDistributed(Conv2D(128, (10, 10), strides=2, padding="same"), batch_input_shape=(None,) + shape))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(64, (6, 6), strides=2, padding="same")))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    # # # # #
    seq.add(TimeDistributed(Conv2DTranspose(128, (6, 6), strides=2, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2DTranspose(3, (11, 11), strides=2, padding="same")))
    # seq.add(LayerNormalization())
    # seq.add(TimeDistributed(Conv2D(3, (11, 11), activation="sigmoid", padding="same")))



    return seq

# test d'un autre autoencoder, non fonctionnel pour l'instant
def autoencoder_modelV2(shape=(5, 112, 112, 3)):
    # Create our convnet with (112, 112, 3) input shape
    convnet_encoder = build_convnet_encoder(shape[1:])
    
    # then create our final model
    model = keras.Sequential()

    model.add(TimeDistributed(convnet_encoder, input_shape = shape))

    # add the convnet with (5, 112, 112, 3) shape
    # model.add(TimeDistributed(convnet_encoder, input_shape=shape))
    # here, you can also use GRU or LSTM
    model.add(GRU(64, return_sequences=True))
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(64, return_sequences=True))
    convnet_decoder = build_convnet_decoder()
    model.add(TimeDistributed(convnet_decoder))
    model.add(TimeDistributed(Conv2D(3, (11, 11), activation="sigmoid", padding="same")))
    return model

# fonction à appeler pour entraîner un model, entraîne uniquement sur les données normales (semi-supervised)
# (à modifier pour rendre plus modulable pour l'instant il faut modifier la fnction si on veut entraîner un autre model)
def train_model():

    X_train, X_test, y_train, y_test = defineTrainSets()



    INSHAPE=(NBFRAME,) + SIZE + (3,) # (5, 112, 112, 3)
    model = autoencoder_model(INSHAPE, 2)

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))

    EPOCHS=20
    # create a "chkp" directory before to run that
    # because ModelCheckpoint will write models inside
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(verbose=1),
        keras.callbacks.ModelCheckpoint(
            'unsupervised_model_normal_only/weights.{epoch:02d}.hdf5',
            verbose=1),
    ]
    model.fit(
        x=X_train,
        y=X_train,
        batch_size=4,
        verbose=1,
        epochs=EPOCHS,
        callbacks=callbacks
    )
# fonction à appeler pour faire des tests avec un model enregistré, sur une série de séquence
# enregistre les images prédites, et renvoie un graphe de la régularité des prédictions au fil des séquences,
# détecte au moins une anomalie dans une suite de séquences d'anomalies,
# mais classifie une partie des séquences des anomalies comme normales
# est-ce le comportement voulu? (on trouve bien une anomalie dans chaque "vidéo" d'anomalie)
def testSample(folderPath, model):
    kModel = keras.models.load_model(model)
    sampleList = []
    imgs = os.listdir(folderPath)
    nbImgs = len(imgs)
    for i in range(nbImgs-NBFRAME):
        sample_frames = []
        for j in range(NBFRAME):
            imgPath = folderPath+'/'+imgs[i+j]
            # loading the image and keeping the target size as (224,224,3)
            img = image.load_img(imgPath, target_size=(112,112,3))
            # img.show()
            # converting it to array
            img = image.img_to_array(img)
            # normalizing the pixel value
            img = img/255
            if not os.path.isdir("./original_imgs/" + folderPath):
                os.makedirs("./original_imgs/" + folderPath)
            image.save_img("./original_imgs/" + folderPath+'/' +str(i)+'_'+str(j)+'.jpeg', img)
            # appending the image to the train_image list
            sample_frames.append(img)
        sampleList.append(sample_frames)
    sampleList = np.array(sampleList)
    predictions = kModel.predict(sampleList, batch_size = 4)
    for i,seq in enumerate(predictions):
        for j,img in enumerate(seq):
            if not os.path.isdir("./predicted_imgs/" + folderPath):
                os.makedirs("./predicted_imgs/" + folderPath)
            image.save_img("./predicted_imgs/" + folderPath +'/'+ str(i)+'_'+str(j)+'.jpeg', img)
    n=len(predictions)
    sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sampleList[i],predictions[i])) for i in range(0,n)])
    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    sr = 1.0 - sa

    # plot the regularity scores
    plt.plot(sr)
    plt.ylabel('regularity score Sr(t)')
    plt.xlabel('frame t')
    plt.show()
    return predictions

# expérimentation avec l'optical flow pour utliser en données d'entraînement, pas testé encore,
# mais données obtenues pas très propres
def optical_flow(folderPath):
    imgs = os.listdir(folderPath)
    nbImgs = len(imgs)
    for i in range(nbImgs-1):
        sample_frames = []
        imgPath = folderPath+'/'+imgs[i]
        imgprev = cv2.imread(imgPath, 1)
        hsv = np.zeros_like(imgprev)
        hsv[...,1] = 255
        imgprev = cv2.cvtColor(imgprev,cv2.COLOR_BGR2GRAY)
        imgPath = folderPath+'/'+imgs[i+1]
        imgnext = cv2.imread(imgPath, 1)
        imgnext = cv2.cvtColor(imgnext,cv2.COLOR_BGR2GRAY)
        

        flow = cv2.calcOpticalFlowFarneback(imgprev,imgnext, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        if not os.path.isdir("./optical_flow/" + folderPath):
                os.makedirs("./optical_flow/" + folderPath)
        cv2.imwrite("./optical_flow/" + folderPath+'/' +str(i)+'.jpeg', rgb)

