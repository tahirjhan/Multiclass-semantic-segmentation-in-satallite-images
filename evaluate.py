import tensorflow as tf
import config
from Load_data import get_dataset
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from set_labels import set_labels
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

image_dataset, mask_dataset = get_dataset()
print(image_dataset.shape[0])
print(mask_dataset.shape[0])

# Set labels
labels_cat = set_labels(mask_dataset)
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.3, random_state=42)

# Load the model
#model = tf.keras.models.load_model(config.model_dir)
model = tf.keras.models.load_model('trained_model.hdf5', compile=False)
print('model loaded successfully')
#model = model.load_weights('best_weight.hdf5')


np.set_printoptions(precision = 4)
# IOU
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
y_test_argmax = np.argmax(y_test, axis=3)

# Using built in keras function for IoU
from keras.metrics import MeanIoU

n_classes = 6
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#######################################################################
# Predict on a few images

import random

test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth = y_test_argmax[test_img_number]
# test_img_norm=test_img[:,:,0][:,:,None]
test_img_input = np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()