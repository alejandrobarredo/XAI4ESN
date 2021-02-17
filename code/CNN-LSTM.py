import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import keras.metrics as km

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix

data_dir = '../datasets/UCF11/Videos/'
img_height, img_width = 160, 120
seq_len = 20

target_dic = {}
for i, name in enumerate(os.listdir(data_dir)):
    target_dic[name] = i
classes = list(target_dic.keys())
#  Creating frames from videos


def frames_extraction(video_path):
    frames_list = []

    vidObj = cv2.VideoCapture(video_path)
    # Used as counter variable
    count = 1

    while count <= seq_len:
        success, image = vidObj.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (img_height, img_width))
            image = image.reshape(image.shape[0], image.shape[1], 1)
            frames_list.append(image)
            count += 1
        else:
            print("Defected frame")
            break

    return frames_list


def create_data(input_dir):
    X = []
    Y = []

    classes_list = os.listdir(input_dir)

    for c in classes_list:
        print(c)
        files_list = os.listdir(os.path.join(input_dir, c))
        for f in files_list:
            if f[-4:] == '.avi':
                frames = frames_extraction(os.path.join(os.path.join(input_dir, c), f))
                if len(frames) == seq_len:
                    X.append(frames)

                    y = [0]*len(classes)
                    y[classes.index(c)] = 1
                    Y.append(y)

    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y


X, Y = create_data(data_dir)

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.20,
                                                    shuffle=True,
                                                    random_state=0)

model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=False,
                     data_format="channels_last",
                     input_shape=(seq_len, img_width, img_height, 1)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(len(classes), activation="softmax"))

model.summary()

opt = keras.optimizers.SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=["accuracy"])


model.fit(x=X_train, y=y_train, epochs=10, batch_size=8,
                    shuffle=True, validation_split=0.3,
                    verbose=2)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_pred)
print('Accuracy: ' + str(acc))
