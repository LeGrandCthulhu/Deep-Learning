import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import dl_lib as dl

# Visualisation de l'utilité de la normalisation de données

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1)) # Loading du dataset "make blob"
X[:, 1] *= 10 # Compression du graphique

lim = 10
h = 100
W1 = np.linspace(-lim, lim, h)
W2 = np.linspace(-lim, lim, h)

W11, W22 = np.meshgrid(W1, W2)
W = np.c_[W11.ravel(), W22.ravel()].T

b = 0
A = dl.model(X, W, b)

L = dl.log_loss_norm(A, y, W11)


plt.scatter(X[:,0], X[:,1], c=y, cmap="summer")
plt.show()
plt.contourf(W11, W22, L, 50, cmap="magma")
plt.colorbar()
plt.show()









