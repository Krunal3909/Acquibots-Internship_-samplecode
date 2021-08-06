# Import libraries
import os
from keras import callbacks
import matplotlib
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab import drive
from imageio import imread
from pylab import rcParams
import tensorflow as tf

# Load images
image_path = "Acquibots/ColorClassf/Data set/"
import os
from posixpath import join
def loadImages(path,folder):
    image_files= sorted([os.path.join(path, folder,file)
                        for file in os.listdir(path+folder+'/')
                        if (file.endswith('.jpg') or file.endswith ('.jpeg'))])
    return image_files

# Data analysis
import matplotlib.pyplot as plt
testimage = loadImages(image_path, 'test_low_light')
img= imread(testimage[32])
plt.imshow(img)
testimage[0].shape


# Building CNN Model

from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten
from keras import optimizers
from keras.callbacks import EarlyStopping ,ModelCheckpoint
import sys
cnn_model=Sequential()

cnn_model.add(Conv2D(input_shape=(128, 128, 3), filters=20, kernel_size=4, strides=2, padding='valid',
                         activation='relu',  data_format='channels_last'))

cnn_model.add(Conv2D(filters=15, kernel_size=3, strides=1, padding='valid', activation='relu',
                          data_format='channels_last'))

cnn_model.add(MaxPool2D(pool_size=3, data_format='channels_last'))

cnn_model.add(Conv2D(filters=20, kernel_size=4, strides=2, padding='valid', activation='relu',
                          data_format='channels_last'))

cnn_model.add(MaxPool2D(pool_size=2, data_format='channels_last'))
cnn_model.add(Flatten())
cnn_model.add(Dropout(0.2))

cnn_model.add(Dense(8, activation='softmax'))

cnn_model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.summary()

 # image augmentation with ImageDataGenerator class

from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        image_path,
        color_mode='rgb',
        # All images will be resized to 64x64        
        target_size=(128,128),
        batch_size=16,
        # Since we use binary_crossentropy loss, we need binary labels
        )

es = [EarlyStopping(monitor='loss',
 mode='auto',
 min_delta = 0.0001, 
 patience= 10,
verbose = 1)
]

test_datagen = ImageDataGenerator(rescale=1./255)

history = cnn_model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=200,
    verbose=1,
    callbacks=es ) # check earlystopping in training

# Save model for future uses
cnn_model.save(image_path+'keras_cnn_model.hdf5')

d = train_generator.class_indices

# Test and prediction

Testimage = loadImages(image_path, ' test_bright_light')

import numpy as np
import cv2
x_cv= testimage
x_t=[]
images = []

for x in x_cv:
    img= imread(x)
    img= img[:,:,:3]
    images.append(img)
    width=128
    height=128
    dim= (width,height)
    resized=cv2.resize(img,dim, interpolation= cv2.INTER_AREA)
    x_t.append(resized)
x_t=np.array
x_t=x_t/255


# load trained model for prediction
from keras.models import load_model
import numpy as np

cnn_model= load_model(image_path +'keras_cnn_model.hdf5')

y_pred = cnn_model.predict(x_t)
predict = np.argmax(y_pred,axis=1)
print(predict)


# Plotting Confusion matrix and classification report
 

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator , FormatStrFormatter
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix , Classification_report
from sklearn import svm, datasets

def plot_confusion_matrix(cm, names , title = ' confusion matrix',cmap =plt.cm.Blues):
    plt.imshow(cm , interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks= np.arrange(len(names))
    plt.xticks(tick_marks, names, rotation = 90)
    plt.yticks(tick_marks, names,)
    plt.tight_layout()
    plt.ylabel('true label')
    plt.ylabel('predicted label')

cm= confusion_matrix(y_true, predict)

print('Plot Of of Confusion Matrix')

plt.figure(figsize=(15,10))
plot_confusion_matrix(cm , true_labels)
plt.grid(which='both')
plt.show()

print(Classification_report(y_true,predict))

