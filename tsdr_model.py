import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt

X = []
y = []

train = pd.read_csv('D:\\GTSRB dataset\\Train.csv')
paths = train['Path'].values
classid = train['ClassId'].values

for path in paths:
    img = cv2.imread('D:\\GTSRB dataset\\' + path)
    img = cv2.resize(img, (32, 32))
    X.append(img)

X = np.array(X)
y = to_categorical(classid,43)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=40)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_ = model.fit(X_train,y_train,64,15,validation_data=(X_test,y_test))
model.save('tsdr_model')

plt.figure(0)
plt.plot(model_.history['accuracy'], label='training accuracy')
plt.plot(model_.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.figure(1)
plt.plot(model_.history['loss'], label='training loss')
plt.plot(model_.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()