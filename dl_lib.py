import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import h5py
from tqdm import tqdm

def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
    Z = np.dot(X, W) + b
    A = 1/(1 + np.exp(-Z))
    
    return A

def log_loss(A, y):
    epsilon = 1e-15
    
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

def log_loss_norm(A, y, shape):
    epsilon = 1e-15
    
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon), axis=0).reshape(shape.shape)

def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    
    return (dW, db)

def update(dW, db, W, b, lr):
    W = W - lr * dW
    b = b - lr * db
    
    return (W, b)

def predict(X, W, b, proba = False):
    A = model(X, W, b)
    if proba:
        print(A)
    return A >= 0.5

def neurone(X, y, X_test=0, y_test=0, lr = 0.1, nbr_iteration = 100):
    W, b = initialisation(X)
    
    history = []
    Loss = []
    Loss_test = []
    acc = []
    acc_test = []
    
    # Apprentissage
    for i in tqdm(range(nbr_iteration)):
        
        # Activations
        A = model(X, W, b)
        
        if i % 10 == 0:
            # Train
            Loss.append(log_loss(A, y))
            y_pred = predict(X, W, b)
            acc.append(accuracy_score(y, y_pred))
            
            if X_test != 0 and y_test != 0:
                # Test
                A_test = model(X_test, W, b)
            
                Loss_test.append(log_loss(A_test, y_test))
                y_pred = predict(X_test, W, b)
                acc_test.append(accuracy_score(y_test, y_pred))

        
        # MAJ
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, lr)
        
        # Historique
        history.append([W, b, Loss, i])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)         
    plt.plot(Loss, label="Train Loss")
    plt.plot(Loss_test, label="Test Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(acc, label="Train Acc")
    plt.plot(acc_test, label="Test Acc")
    plt.legend()
    
    plt.show()
    
    return (W, b)

def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test