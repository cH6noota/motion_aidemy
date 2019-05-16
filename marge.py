import numpy as np


def read(name):
	return np.load(name)
data0 = read("./correct_dataset/X.npy")
data1 = read("./uncorrect_dataset/faces.npy")
data2 = read("./uncorrect_dataset/CIFAR-10.npy")

marge=np.concatenate([data0, data1, data2])

np.save('/Users/moritachikara/Desktop/motion/dataset/X.npy',marge)
