import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import dl_lib as dl

# Variables
new_plant = np.array([2, 1]) # Variable de prédiction

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1)) # Loading du dataset "make blob"


# Calcul et fonctions
W, b = dl.neurone(X, y) # initialisation, entrainement et sorti du vecteur W et de b
print(dl.predict(new_plant, W, b, True)) # Prédiction de la variable


# Tracé
x0 = np.linspace(-1, 4, 100)
x1 = (-W[0] * x0 - b) / W[1]

plt.scatter(X[:,0], X[:,1], c=y, cmap="summer") # Dataset
plt.scatter(new_plant[0], new_plant[1], c="r") # Variable de prédiction
plt.plot(x0, x1, c="orange", lw=3) # Frontière de décision
plt.show()
        






    
    
    
    
    