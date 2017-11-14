import argparse
import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Lambda, Dense, Flatten, Cropping2D, Convolution2D, Dropout
import matplotlib.pyplot as plt

from helpFunctions import *
from myModels import *


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parse arguments to model.py')

    parser.add_argument('--modelFile',
                        action ='store',
                        type=str)
    parser.add_argument('--modelId',
                        action ='store',
                        type=int)
    parser.add_argument('--newModel',
                        action ='store_true',
                        default=False)
    #model = createNVidiaModel()
    #model.compile(loss='mse', optimizer='adam')

    args = parser.parse_args()
    
    if args.newModel:
        model = createNVidiaModel()
        model.compile(loss='mse', optimizer='adam')
    else:
        model = load_model(args.modelFile)

    dataDir = './fastDriveData/'

    samples = getCsvLines(dataDir+'/driving_log.csv')
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    history_object=model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=1)
    model.save(args.modelFile)

    # plot loss validation and training
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    print(model.summary())

