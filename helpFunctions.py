import argparse
import os
import csv
import cv2
import numpy as np
import sklearn

def getCsvLines(path):
    # read image paths and corresponding steering angles
    samples =[]
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for i_line in reader:
            samples.append(i_line)
    return samples


def prepareTrainingData(path, useAllCameraImages=False, cameraCorrectionFactor = 0.15):
    imagePaths = []
    angles = []
    camera_angle_factor = [0,+1,-1]

    samplesLines = getCsvLines(path)
    if useAllCameraImages:
        imagesToUse = 3
    else:
        imagesToUse = 1

    for i_line in samplesLines:

        for i in range(imagesToUse):
            i_imagePath = i_line[i].split('/')[-1]
            imagePaths.append(i_imagePath)
            i_angle = float(i_line[3])+cameraCorrectionFactor*camera_angle_factor[i]
            angles.append(i_angle)
        

    samples = list(zip(imagePaths, angles))
    return samples

def generator(samples, dataDir, batch_size=32,color_space='RGB', useFlipImages=False):
    num_samples = len(samples)
    baseDir = dataDir +'/IMG/'

    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images, angles = [], []
            for i_imagePath, i_angle in batch_samples:
                
                #images.append(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB))
                i_image = cv2.imread(baseDir+i_imagePath)
                #print(name)
                # crop the image
                i_image = i_image[51:140,:]
                
                # resize the image to 66 by 200
                i_image = cv2.resize(i_image, (200, 66))
                
                if color_space == 'RGB':
                    i_image = cv2.cvtColor(i_image, cv2.COLOR_BGR2RGB)
                elif color_space == 'HSV':
                    i_image = cv2.cvtColor(i_image, cv2.COLOR_BGR2HSV)
                elif color_space == 'LUV':
                    i_image = cv2.cvtColor(i_image, cv2.COLOR_BGR2LUV)
                elif color_space == 'HLS':
                    i_image = cv2.cvtColor(i_image, cv2.COLOR_BGR2HLS)
                elif color_space == 'YUV':
                    i_image = cv2.cvtColor(i_image, cv2.COLOR_BGR2YUV)
                elif color_space == 'YCrCb':
                    i_image = cv2.cvtColor(i_image, cv2.COLOR_BGR2YCrCb)

                        
                images.append(i_image)
                # add steering angle according to which camera it came from i.e. center, left or right
                angles.append(i_angle)
                
                if useFlipImages:
                    # add flipped images also
                    images.append(np.fliplr(i_image))
                    angles.append(-i_angle)


                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                #print(X_train.size,y_train.size)
                yield sklearn.utils.shuffle(X_train, y_train)
