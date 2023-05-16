import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import dl_lib as dl
import plotly.graph_objects as go 

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

W, b = dl.neurone(X, y, learningRate=0.2, nCycles=250)

print(dl.prediction(np.array([2,1]), W, b))

x0 = np.linspace(-1, 4, 100)
x1 = (-W[0] * x0 - b) / W[1] 
plt.scatter(X[:,0], X[:, 1], c=y, cmap="summer")
plt.scatter(2, 1, c="r")
plt.plot(x0, x1, c="orange")
plt.show()

        






    
    
    
    
    