from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D

import sys
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.python import keras
from keras import layers, models, regularizers
from keras.models import Sequential
import os
from matplotlib import pyplot as plt
import numpy as np
import random

train_data_dir = 'FER-2013-dataset/train/'
validation_data_dir = 'FER-2013-dataset/test/'

IMG_HEIGHT = 48
IMG_WIDTH = 48
batch_size = 200
print("------Loading Images------")
# train_datagen = ImageDataGenerator(
# 					rescale=1./255,
# 					rotation_range=30,
# 					shear_range=0.3,
# 					zoom_range=0.3,
# 					horizontal_flip=True,
# 					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
# 					train_data_dir,
# 					color_mode='grayscale',
# 					target_size=(IMG_HEIGHT, IMG_WIDTH),
# 					batch_size=batch_size,
# 					class_mode='categorical',
# 					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(IMG_HEIGHT, IMG_WIDTH),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)

# Verify our generator by plotting a few faces and printing corresponding labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

img, label = validation_generator.__next__()


# i = random.randint(0, (img.shape[0])-1)
# image = img[i]
# labl = class_labels[label[i].argmax()]
# plt.imshow(image[:,:,0], cmap='gray')
# plt.title(labl)
# plt.show()

# train_path = "/content/gdrive/MyDrive/Emotion Recognition/FER-2013-dataset/train/"
test_path = "/content/gdrive/MyDrive/Emotion Recognition/FER-2013-dataset/test/"

# num_train_imgs = 0
#
# for root, dirs, files in os.walk(train_path):
# 	num_train_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
	num_test_imgs += len(files)

# Test the model
my_model = load_model('Models/ven_model_20epochs_final_v6.h5', compile=False)

# Generate a batch of images
test_img, test_lbl = validation_generator.__next__()
predictions=my_model.predict(test_img)

predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_lbl, axis=1)

from sklearn import metrics

print("Accuracy = ", metrics.accuracy_score(test_labels, predictions))

# Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, predictions)
# print(cm)
import seaborn as sns
sns.heatmap(cm, annot=True)
plt.show()

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# Check results on a few select images
n = random.randint(0, test_img.shape[0] - 1)
image = test_img[n]
orig_labl = class_labels[test_labels[n]]
pred_labl = class_labels[predictions[n]]
plt.imshow(image[:,:,0], cmap='gray')
plt.title("Original label is:"+orig_labl+" Predicted is: " + pred_labl)
plt.show()
