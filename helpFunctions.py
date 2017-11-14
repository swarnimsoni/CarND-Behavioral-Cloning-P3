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


def generator(samples,dataDir, batch_size=32, useFlipImages=False, useAllCameraImages=False, cameraCorrectionFactor = 0.15):
    num_samples = len(samples)
    baseDir = dataDir +'/IMG/'

    cameraCorrectionFactor = 0.15
    camera_angle_factor = [0,+1,-1]
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images, angles = [], []
            for batch_sample in batch_samples:

                if useAllCameraImages:
                    imagesToUse = 3
                else:
                    imagesToUse = 1

                for i in range(imagesToUse):
                    name = baseDir+batch_sample[i].split('/')[-1]
                    #images.append(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB))
                    i_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    images.append(i_image)
                    # add steering angle according to which camera it came from i.e. center, left or right
                    i_angle = float(i_line[3])+cameraCorrectionFactor*camera_angle_factor[i]
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
