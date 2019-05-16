import numpy as np
import cv2
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical


pred=np.argmax(model.predict([[img]]))

#モデルの定義、学習部分
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

input_tensor = Input(shape=(32, 32, 3))
#--------------------------------------------------------
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='sigmoid'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2, activation='softmax'))
# vgg16とtop_modelを連結
model = Model(input=vgg16.input, output=top_model(vgg16.output))
# 15層目までの重みを固定
for layer in model.layers[:15]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


model.fit(X_train, y_train,   validation_data = (X_test, y_test) )
model.save('model_kadai.h5')


