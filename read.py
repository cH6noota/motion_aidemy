import numpy as np


def read(name):
	return np.load(name)

data2 = read("/Users/moritachikara/pywork/good_box/2296X.npy")
#data2 = data2[:,:,:, np.newaxis]
#print(data2[0])
print(data2.shape)

