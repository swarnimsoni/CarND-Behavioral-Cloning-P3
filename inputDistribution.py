# plot histogram to find out distribution of training data, steering angle vs # of samples


import csv
import numpy as np
import matplotlib.pyplot as plt

from helpFunctions import *

filePath = './myCapturedData/driving_log.csv'
lines=getCsvLines(filePath) 
steeringAngles=[]

baseDir = './data/IMG/'
camera_correction_factor = 0.12
camera_angle_factor = [0,+1,-1]
for i_line in lines:        
    # use images from all cameras, center, left and right
    for i in range(3):
        #print(i_line)
        # add steering angle according to which camera it came from i.e. center, left or right
        i_angle = float(i_line[3])+camera_correction_factor*camera_angle_factor[i]
        steeringAngles.append(i_angle)
        steeringAngles.append(-i_angle)
#convert images and steering angles into numpy arrays
y_train = np.array(steeringAngles)

plt.hist(y_train, 101, facecolor='b', alpha=0.75)
plt.title('Histogram of training set')
plt.xlabel('Steering Angles')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
