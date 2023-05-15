import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def initialisation(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def modele(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def cout(A: np.ndarray, y: np.ndarray) -> float:
    return -1 / len(y) * np.sum(y * np.log(A) + (1-y) * np.log(1 - A))

def gradients(A: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.float64]:
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)

def maj(dW: np.ndarray, db: np.float64, W: np.ndarray, b: np.ndarray, learningRate: float) -> tuple[np.ndarray, np.ndarray]:
    W = W - learningRate * dW
    b = b - learningRate * db
    return (W, b)

def prediction(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple[bool, np.ndarray]:
    A = modele(X, W, b)
    return (A >= 0.5, A)

def neurone(X: np.ndarray, y: np.ndarray, learningRate: float=0.1, nCycles: int=100):
    # init W, b
    W, b = initialisation(X)

    Cout = []

    for i in range(nCycles):
        A = modele(X, W, b)
        Cout.append(cout(A, y))
        dW, db = gradients(A, X, y)
        W, b = maj(dW, db, W, b, learningRate)
    
    y_predict = prediction(X, W, b)[0] 
    print(accuracy_score(y, y_predict)) # permet d'évaluer son niveau sur un dataset

    #plt.plot(Cout) # Montre la fonction cout
    #plt.show()

    return (W, b)
