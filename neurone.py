import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import dl_lib as dl

lim = 10
h = 100
W1 = np.linspace(-lim, lim, h)
W2 = np.linspace(-lim, lim, h)

W11, W22 = np.meshgrid(W1, W2)

Wf = np.c_[W11.ravel(), W22.ravel()].T

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

W, b = dl.neurone(X, y, learningRate=0.1, nCycles=100)

print(dl.prediction(np.array([2,1]), W, b))

b = 0
A = dl.modele(X, Wf, b)
epsilon = 1e-15
L = -1 / len(y) * np.sum(y * np.log(A + epsilon) + (1-y) * np.log(1 - A + epsilon), axis=0).reshape(W11.shape)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.contourf(W11, W22, L, 20, cmap="magma")
plt.colorbar()

plt.subplot(1, 2, 2)
x0 = np.linspace(-1, 4, 100)
x1 = (-W[0] * x0 - b) / W[1] 
plt.scatter(X[:,0], X[:, 1], c=y, cmap="summer")
plt.scatter(2, 1, c="r")
plt.plot(x0, x1, c="orange")
plt.show()

        






    
    
    
    
    