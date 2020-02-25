#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

# just to overwrite default colab style
plt.style.use('default')
plt.style.use('seaborn-talk')


# Linear Regression using Stochastic Gradient Descent
def batchloader(X, Y, batchsize = 20):
    n = Y.shape[0]
    idx = np.random.choice(np.arange(n),size=batchsize,replace=False)
    X_batch = X[idx,:]
    Y_batch = Y[idx,:]
    return X_batch, Y_batch

def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)

def crossEntropy(X, Y, theta):
    fce = 0
    
    n, m = X.shape
    n, p = Y.shape 
    
    for i in range(n):
        Y_pred = softmax(np.dot(X[i],theta))
        fce += np.dot(Y[i].T, np.log(Y_pred))  
    fce *= (-1/n)
    return fce
    
def SGD(X, Y, learning_rate=0.001, epochs=100, bs = 0.2, alpha = 0.2):
    
    n, m = X.shape
    n, p = Y.shape
    
    w = np.random.randn(m,p)
    b = np.random.randn(1,p)
    
    batchsize = round(bs*n)
    
    # preprocessing Data
    X = np.append(X, np.ones((n,1)), axis=1)
    
    n, mpn = X.shape

    COST = np.zeros(epochs)
    theta = np.append(w, b, axis=0)
    
    for i in range(epochs):
    
        # Get Batch 
        X_batch, Y_batch = batchloader(X, Y, batchsize) 
        Y_pred = softmax(np.dot(X_batch,theta))
        
        # perform gradient descent
        gradJ = (1/n)*( np.dot(X_batch.T, (Y_pred - Y_batch) )) 
        theta = theta - learning_rate * gradJ - 2*alpha*np.append(w,np.zeros((1,p)),axis=0)
         
        w = theta[:m,:p]    

        # get cost
        COST[i] = crossEntropy(X_batch, Y_batch, theta)
    
    return theta, COST 
    
def accuracy(Y_pred, Y):
    n,p = Y.shape
    acc = np.sum([np.argmax(Y_pred[i])==np.argmax(Y[i]) for i in range(n)])/(0.01*n)
    return acc


# In[97]:


a = np.array([[1,2,3,4], [4,5,6,7]])
print([np.argmax(a[i])==3 for i in range(2)])
print(a[1])


# In[ ]:


# Load data
Xtr = np.load("mnist_train_images.npy")
n = Xtr.shape[0]
Xtr = Xtr.reshape((n,-1))
Ytr = np.load("mnist_train_labels.npy")

# Get Validation Set
Xv = np.load("mnist_validation_images.npy")
nv = Xv.shape[0]
Xv = Xv.reshape((nv,-1)) # feature vector is row vector 
Yv = np.load("mnist_validation_labels.npy")

# preprocessing on validation set 
Xv = np.append(Xv, np.ones((nv,1)), axis=1)

# print(Xtr.shape, Ytr.shape, Xv.shape, Yv.shape)

# Tune Hyper parameter
LR = [0.001, 0.005, 0.01, 0.05]
EPOCHS = [50, 100, 200, 400]
BATCHSIZE = [0.1, 0.2, 0.3, 0.5]
ALPHA = [0.001, 0.002, 0.005, 0.01]

cost = 1000000

iter = 0

for lr_ in LR:
    for epochs_ in EPOCHS:
        for bs_ in BATCHSIZE:
            for alpha_ in ALPHA:
                theta, COST = SGD(Xtr, Ytr, learning_rate=lr_, epochs=epochs_, bs=bs_, alpha=alpha_) 
                
                Y_pred = softmax(np.dot(Xv, theta))
                accu = accuracy(Y_pred, Yv)
                loss = COST[-1]
                print("accracy: ", accu)
                print("iter: ", iter, ", loss: ", COST[-1])
                iter += 1 
                
                if(loss<cost):
                    lr = lr_
                    epochs = epochs_
                    bs = bs_
                    alpha = alpha_
                    cost=loss

# Training on tuned hyperparameters
theta, COST = SGD(Xtr, Ytr, learning_rate=lr, epochs=epochs, bs=bs, alpha=alpha)

plt.plot(COST)
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.title('Stochastic Gradient Descent with L2 Regularization')
plt.show()

# Testing 
X_te = np.load("mnist_test_images.npy")
n = X_te.shape[0]
X_te = X_te.reshape((n,-1))

# preprocessing Data
X_te = np.append(X_te, np.ones((n,1)), axis=1)

yte = np.load("mnist_test_labels.npy")
MSE_test = crossEntropy(X_te, yte, theta)
Y_pred = softmax(np.dot(X_te, theta))
accu = accuracy(Y_pred, yte)
print('accuracy', accu)
print('MSE on test data: ', MSE_test[0,0])
print('Tuned Hyperparameters: ')
print('learning rate: ', lr, ', Epochs: ', epochs, ', mini-batchsize (in %): ', bs*100, ', alpha: ', alpha)

