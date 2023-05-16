import numpy as np
import matplotlib.pyplot as plt
import dl_lib as dl

X_train, y_train, X_test, y_test = dl.load_data()
X_train_r = X_train.reshape(X_train.shape[0], -1) / X_train.max()
X_test_r = X_test.reshape(X_test.shape[0], -1) / X_train.max()

W, b = dl.neurone(X_train_r, y_train, learningRate=0.3, nCycles=10000)










