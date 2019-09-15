import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def csvToImg(csv_input,file_name):
    #create an image
    imar = np.array(csv_input).transpose()
    plt.imsave(file_name, imar)

    # read the image
    im = plt.imread('%s.jpg'%file_name)
    # show the image
    plt.imshow(im)
    plt.show()

    #save the image array to binary file
    np.save(file_name, im)
    # load the image from binary file
    new_im= np.load( '%s.npy' %file_name)
    # show the loaded image
    plt.imshow(new_im)
    plt.show()

'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
#
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense
# from keras import backend as K
#
#
# # dimensions of our images.
# img_width, img_height = 150, 150
#
# #Todo: complete this
# train_handstart_dir = 'data/train'
# train_firstdigittouch_dir = 'data/train'
# # Todo: build binary test folder for HandStart,FirstDigitTouch
# test_data_dir = 'data/validation'
# nb_train_samples = 2000
# nb_validation_samples = 800
# epochs = 50
# batch_size = 16


import os, cv2, re, random
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import pdb;
from keras import optimizers

x = np.ones(shape=(150,34))


import numpy as np

x = np.ones(shape=(149, 34))

def triple_square_padding(x, size=150):

    to_add_x = size - x.shape[0]
    z = np.zeros(shape=(to_add_x, x.shape[1]))
    new_x = np.concatenate((x, z), axis=0)

    to_add_y = size - new_x.shape[1]
    z = np.zeros(shape=(new_x.shape[0], to_add_y))
    new_x = np.concatenate((new_x, z), axis=1)
    new_x = new_x.flatten()
    #new_x = np.array([new_x, new_x, new_x])
    #print(new_x.get(0))
    return np.transpose(new_x)

def format_to_3d_img(df):
	X = np.array(df)
	return np.reshape(X, (X.shape[0], 1, X.shape[1]))

def prepare_data(list_of_images):
    """
    Returns two arrays:
        x is an array of resized images
        y is an array of labels
    """
    x = []  # images as arrays
    y = []  # labels

    for image in list_of_images:
        img_read = np.array(pd.read_csv(image).values)
        x.append(triple_square_padding(img_read))
    for i in list_of_images:
        if 'FirstDigitTouch' in i:
            y.append(1)
        elif 'HandStart' in i:
            y.append(0)
        else:
            print('neither cat nor dog name present in images')

    return x, y

img_width = 150 #65??
img_height = 150

TRAIN_HandStart_DIR = 'grasp-and-lift-eeg-detection/my_GAL_dataset/HandStart'
TRAIN_FirstDigitTouch_DIR = 'grasp-and-lift-eeg-detection/my_GAL_dataset/FirstDigitTouch'

#TEST_DIR = 'datasets/cats and dogs/test/'
train_images = [TRAIN_HandStart_DIR + "/" + i for i in os.listdir(TRAIN_HandStart_DIR)] # use this for full dataset
train_images+= [TRAIN_FirstDigitTouch_DIR + "/" + i for i in os.listdir(TRAIN_FirstDigitTouch_DIR)]

X, Y = prepare_data(train_images)

X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=1)

# train = 0.8* 0.8 = 0.64
# test = 0.8*0.2 = 0.16
X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train, test_size=0.2, random_state=1)

#import pdb;
#pdb.set_trace();

nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()


#we tried: 32 layers, 64 layers,



sgd = optimizers.SGD(lr=0.001, decay=1e-6, nesterov=True)

model.add(Dense(1024, activation='sigmoid',kernel_initializer='uniform',  input_dim=22500))
model.add(Dropout(0.2))
model.add(Dense(2048, activation='sigmoid',kernel_initializer='uniform',  input_dim=22500))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='mean_squared_error', optimizer=sgd)

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#model.compile(optimizer='rmsprop',
 #             loss='binary_crossentropy',
 #             metrics=['accuracy'])

# print summary..
model.summary()
#
# # this is the augmentation configuration we will use for testing:
# # only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)
print('-----1-----')
#
#
# # Augmentation:

# # this is the augmentation configuration we will use for training:
#

#
# # Prepare generators for training and validation sets:
#


model.fit(np.array(X_train), np.array(Y_train), epochs=20, batch_size=32)

# loss_history = LossHistory()
# lrate = LearningRateScheduler(step_decay)
# callbacks_list = [loss_history, lrate]
#
# history = model.fit(X_train, y_train,
#    validation_data=(X_test, y_test),
#    epochs=epochs,
#    batch_size=batch_size,
#    callbacks=callbacks_list,
#    verbose=2)

# Saving the model

model.save_weights('model_wieghts_MLP.h5')
model.save('model_keras_MLP.h5')


print('-----2-----')
# Now the Loading and predicting phase:
#X_test.append(X_train[0])

prediction_probabilities = model.predict(np.array(X_test), verbose=1,steps=1)

# Generate .csv for submission

# test_images_dogs_cats
counter = range(1, len(X_test) + 1)
solution = pd.DataFrame({"id": counter, "label":list(prediction_probabilities)})
cols = ['label']

for col in cols:
    solution[col] = solution[col].map(lambda x: str(x).lstrip('[').rstrip(']')).astype(float)

solution.to_csv("gal_MLP_fit_two_label.csv", index = False)


