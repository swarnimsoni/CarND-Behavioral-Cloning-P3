# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nVidia_model.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/sampleImage.jpg "Sample Image"
[image4]: ./examples/flippedImage.jpg "Flipped Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/sampleImage.jpg "Normal Image"
[image7]: ./examples/flippedImage.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
My main code file is model.py, which is used to build and train the model.
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

##### Using generators

The generators turned out to pragmatic solution to prevent my machine running out of RAM. The generators read images saved by simulator, shuffles them randomly, uses images from all three cameras, apply angle correction, flip each original image and then yield them.

### Model Architecture and Training Strategy

#### 1. nVidia's model architecture has been employed

I have tried to reproduce model from  Nvidia (in model.py in function createNvidiaModel()) 

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


The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py). Dropout rate of 50% is used.

The data set was divided into two parts, training set (80%) and validation set (20%). Loss on validation set proved to be essential to remove over/under fitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used only center lane driving.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia's network. It was suggested in the forums, and I decided to give it a try. It turns out, this model performs really well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

Initially I missed to use activations for the conlutional layers, and my model was not performing well.

My first working model was overfitting (validation loss was high). To combat the overfitting, I modified the model and introduced dropout layers after each convolution layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, e.g. when the car reaches the first bridge.

After each training session uses the model parameters saved in previous session (saved in model.h5). This was I was able to fine tune the paramters, rather than train the network from scratch.

At the end of the process, the vehicle is able to drive autonomously around the track 1 without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture:

![Modified Nvidia Network][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![sample image][image3]

[//]: I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I also flipped images and angles thinking that this would remove model's bias towards turning left almost all the time.

For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

To increase the size of dataset without the need to record extra laps, I used images from left and right cameras as well. After, using all cameras and flipping every image, my dataset grew 6 times the original dataset.

After the collection process, I had 11226 number of data points/(image,angle) pairs (3742 left, 3742 right and 3742 center images) . I then preprocessed this data by cropping top 50 pixels (containing terrain, sky etc. irrelavant info to train model) and bottom 25 pixels (containing hood of the car).

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I have trained the model in many sessions, each session ranging from 2 epochs to 3. This iterative approach was used to figure out, where my model is lacking after each iterations and relevant modifications were done in the code. Another reason for choosing to train in small sessions was limited computing resources. I trained the model on a machine with 8 GB ram and i7 Processor, without a GPU. So I couldn't wait for long time, running each session for large number of epochs.

Overall I must have run my model for 30 epochs in total, however, each session had different parameters.

I used an adam optimizer so that manually training the learning rate wasn't necessary.