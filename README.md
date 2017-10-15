# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


[image1]: ./examples/nVidia_model.png "Model Visualization"
[image2]: ./examples/sampleImage.jpg "A sample image from training set"
[image6]: ./examples/sampleImage.jpg "Normal Image"
[image7]: ./examples/flippedImage.jpg "Flipped Image"

Introduction
---


The objective of this project was to apply deep learning skill/tools to build a model that can drive on road (in simulator) without human intervention.
I have to admit, this project turns out to be most satisfying project so far. Watching my trained model to smoothly go over the track, was enjoyable.

This repository contains starting files for the Behavioral Cloning Project.

I have successfully build and trained a deep learning model using Keras, that can drive autonomously in the simulator.

Files submitted
---

To meet specifications, the project includes following five files: 
* model.py 
* drive.py 
* model.h5 (a trained Keras model)
* a report writeup file (writeup.md)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)


Model architecture
---
As suggested in video lectures, I started with a powerful architecture i.e. Nvidia architecture. Following is exact details of each layer and its type:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:|
| Input                 | 66x200x3 BRG image                             |
| Normalization         | pixel varies between [-0.5,0.5]                             |
| Convolution 5x5       | 2x2 stride, valid padding, outputs 31x98x24   |
| RELU                  |                                               |
| Droput                | 0.5                                          |
| Convolution 5x5       | 2x2 stride, valid padding, outputs 14x47x36   |
| RELU                  |                                               |
| Droput                | 0.5                                          |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 5x22x48   |
| RELU                  |                                               |
| Droput                | 0.5                                          |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 3x20x64   |
| RELU                  |                                               |
| Droput                | 0.5                                          |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 1x18x64   |
| RELU                  |                                               |
| Droput                | 0.5                                          |
| Flatten               | outputs 1152                                   |
| Fully Connected Layer | outputs 1164                                  |
| Fully Connected Layer | outputs 100                                  |
| Fully Connected Layer | outputs 50                                  |
| Fully Connected Layer | outputs 1                                 |

![Model architecture][image1]

Training the model
---

![Sample image from training set][image2]

To augment the data set, I also flipped images and angles thinking that this would remove model's bias towards turning left almost all the time.

For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I have trained the model in multiple iterations, and each iteration uses model saved in previous iteration. Generators were great help to prevent my machine to run out of memory.

I generated additional data by using images from all cameras and flipping each image. Flipping each image (and negating its corresponding angle) prevented the model to be baised towards turning left.

Adam optimizer is used, so I didn't have to worry about learning rate (one hyper-parameter less to tune).
