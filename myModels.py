# replicate nVidia pipeline
def createNVidiaModel(activation_function='elu', dropOutRate=0.0):

    model = Sequential()

    # add normalization and cropping layers
    model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape = (160,320,3)))

    # resize images to 66*200
    #    model.add(Lambda(lambda x:
    # normalise the input images so that input values to NN varies from -0.5 to 0.5    
    model.add(Lambda(lambda x: x/255-0.5))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation=activation_function))
    model.add(Dropout(dropOutRate))    
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation=activation_function))
    model.add(Dropout(dropOutRate))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation=activation_function))
    model.add(Dropout(dropOutRate))
    model.add(Convolution2D(64,3,3, subsample=(1,1), activation=activation_function))
    model.add(Dropout(dropOutRate))
    model.add(Convolution2D(64,3,3, subsample=(1,1), activation=activation_function))
    model.add(Dropout(dropOutRate))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model


def smallerNVidiaModel(activation_function='elu', dropOutRate=0.0):

    model = Sequential()

    # add normalization and cropping layers
    model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape = (160,320,3)))

    # resize images to 66*200
    #    model.add(Lambda(lambda x:
    # normalise the input images so that input values to NN varies from -0.5 to 0.5    
    model.add(Lambda(lambda x: x/255-0.5))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation=activation_function))
    model.add(Dropout(dropOutRate))    
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation=activation_function))
    model.add(Dropout(dropOutRate))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation=activation_function))
    model.add(Dropout(dropOutRate))
    model.add(Convolution2D(64,3,3, subsample=(1,1), activation=activation_function))
    model.add(Dropout(dropOutRate))
    model.add(Convolution2D(64,3,3, subsample=(1,1), activation=activation_function))
    model.add(Dropout(dropOutRate))
    model.add(Flatten())
#    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model
