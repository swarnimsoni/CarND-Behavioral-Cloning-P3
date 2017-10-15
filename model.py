import os
import csv
import cv2
import numpy as np
import sklearn

samples=[]

# read image paths and corresponding steering angles
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i_line in reader:
        samples.append(i_line)
        #print(lines[0])

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

        
# TODO: pre-allocate memory for steering angles and images, to improve memory effic
steeringAngles=[]
images = []
baseDir = './data/IMG/'
camera_correction_factor = 0.15
camera_angle_factor = [0,+1,-1]
#for i_line in samples:        
#    # use images from all cameras, center, left and right
#    for i in range(3):
#        images.append(cv2.cvtColor(cv2.imread(baseDir+i_line[i].split('/')[-1]), cv2.COLOR_BGR2RGB))
#        # add steering angle according to which camera it came from i.e. center, left or right
#        steeringAngles.append(float(i_line[3])+camera_correction_factor*camera_angle_factor[i])


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images, angles = [], []
            for batch_sample in batch_samples:

                for i in range(3):
                    name = baseDir+batch_sample[i].split('/')[-1]
                    #images.append(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB))
                    images.append(cv2.imread(name))
                    # add steering angle according to which camera it came from i.e. center, left or right
                    angles.append(float(i_line[3])+camera_correction_factor*camera_angle_factor[i])

                    # add flipped images also
                    
                    
                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                #print(X_train.size,y_train.size)
                yield sklearn.utils.shuffle(X_train, y_train)
        
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#convert images and steering angles into numpy arrays
#X_train = np.array(images)
#y_train = np.array(steeringAngles)

# build the neural network
from keras.models import Sequential, load_model
from keras.layers import Lambda, Dense, Flatten, Cropping2D, Convolution2D, Dropout
import matplotlib.pyplot as plt

# replicate nVidia pipeline
def createNVidiaModel(dropOutRate=0.5):

    model = Sequential()

    # add normalization and cropping layers
    model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape = (160,320,3)))

    # resize images to 66*200
    #    model.add(Lambda(lambda x:
    # normalise the input images so that input values to NN varies from -0.5 to 0.5    
    model.add(Lambda(lambda x: x/255-0.5))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(dropOutRate))    
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(dropOutRate))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Dropout(dropOutRate))
    model.add(Convolution2D(64,3,3, subsample=(1,1), activation='relu'))
    model.add(Dropout(dropOutRate))
    model.add(Convolution2D(64,3,3, subsample=(1,1), activation='relu'))
    model.add(Dropout(dropOutRate))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model


#model = Sequential()

# TODO: add image generators
# TODO: flip images

# TODO: crop images, use lowest reolution images also
#60 pixels from top
#25 pixels from below
# before cropping, image size = (160,320)
# after cropping,  image size = (135,260)
#model.add(Cropping2D(cropping=((60,25),(0,0)), input_shape = (160,320,3)))
# normalise the input images so that input values to NN varies from -0.5 to 0.5
#model.add(Lambda(lambda x: x/255-0.5, input_shape = (160,320,3)))
#model.add(Lambda(lambda x: x/255-0.5))
#model.add(Flatten())
#model.add(Dense(1))

model = createNVidiaModel()
model.compile(loss='mse', optimizer='adam')
#model = load_model('model.h5')
#history_object= model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5,verbose=1)
history_object=model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)


# plot loss
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
print(model.summary())
model.save('model.h5')