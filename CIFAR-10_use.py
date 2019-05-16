import keras
from keras.datasets import cifar10
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

data=X_train[:1300]

for i in range(1300):
	pilImg = Image.fromarray(np.uint8(X_train[i]))
	pilImg = pilImg.resize((64,64))
	pilImg.save("./CI_box/"+str(i)+".jpg")


#pilImg = Image.fromarray(np.uint8(X_train[0]))
#pilImg.save("testtest.jpg")
#img_resize = Image.open("testtest.jpg").resize((64,64))
#img_resize.save("testtest2.jpg")



