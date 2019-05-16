from PIL import Image
import glob
import numpy as np
from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img,img_to_array

temp_img = Image.open("/Users/moritachikara/Desktop/test.jpg").resize((64,64))
temp_img = temp_img.convert('L')
temp_img_array = np.asarray(temp_img)
temp_img_array =temp_img_array [:,:,np.newaxis]

model = load_model('testmodel.h5')
pred=np.argmax(model.predict([[temp_img_array]]))
print(pred)


