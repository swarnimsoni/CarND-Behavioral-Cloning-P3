# plot histogram to find out distribution of training data, steering angle vs # of samples


import csv
import numpy as np
import matplotlib.pyplot as plt
lines=[]

# read image paths and corresponding steering angles
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i_line in reader:
        lines.append(i_line)
        #print(lines[0])
steeringAngles=[]

baseDir = './data/IMG/'
camera_correction_factor = 0.15
camera_angle_factor = [0,+1,-1]
for i_line in lines:        
    # use images from all cameras, center, left and right
    for i in range(3):
        # add steering angle according to which camera it came from i.e. center, left or right
        steeringAngles.append(float(i_line[3])+camera_correction_factor*camera_angle_factor[i])

#convert images and steering angles into numpy arrays
y_train = np.array(steeringAngles)

plt.hist(y_train, 101, facecolor='b', alpha=0.75)
plt.title('Histogram of training set')
plt.xlabel('Steering Angles')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
