import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import h5py
from tqdm import tqdm

def initialisation(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def modele(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def cout(A: np.ndarray, y: np.ndarray) -> float:
    epsilon = 1e-15
    return -1 / len(y) * np.sum(y * np.log(A + epsilon) + (1-y) * np.log(1 - A + epsilon))  

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

def neurone(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, learningRate: float=0.1, nCycles: int=100):
    # init W, b
    W, b = initialisation(X)

    trainLoss = []
    trainAccuracy = []
    testLoss = []
    testAccuracy = []

    for i in tqdm(range(nCycles)):
        A = modele(X, W, b)

        if i % 10 == 0:
            trainLoss.append(cout(A, y))
            y_predict = prediction(X, W, b)[0]
            trainAccuracy.append(accuracy_score(y, y_predict))

            A_test = modele(X_test, W, b)
            testLoss.append(cout(A_test, y_test))
            y_predict = prediction(X_test, W, b)[0]
            testAccuracy.append(accuracy_score(y_test, y_predict))

        dW, db = gradients(A, X, y)
        W, b = maj(dW, db, W, b, learningRate)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(trainLoss, label="train")
    plt.plot(testLoss, label="test")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(trainAccuracy, label="train")
    plt.plot(testAccuracy, label="test")
    plt.legend()
    plt.show()

    return (W, b)

def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) 
    y_train = np.array(train_dataset["Y_train"][:]) 

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:])
    y_test = np.array(test_dataset["Y_test"][:])
    
    return X_train, y_train, X_test, y_test