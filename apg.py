#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from sklearn.metrics import accuracy_score
from scipy import io
import os
import math
numpy.random.seed(42)

# Set working directory here with os,chdir('./path')
data = io.loadmat('emaildata.mat')
ytrain = data['ytrain'].T
ytest = data['ytest'].T
xtrain = np.concatenate((data['Xtrain'], np.ones((3065, 1))), axis=1).T
xtest = np.concatenate((data['Xtest'], np.ones((1536, 1))), axis=1).T

def sigmoid(x):
    a = 1/(1+np.exp(-x))
    return a

def prediction(w, Data):
    pred = []
    z = np.dot(w.T, Data)
    a = sigmoid(z)
    for i in range(0, len(a[0])):
        if (a[0][i] > 0.5):
            pred.append(1)
        elif (a[0][i] <= 0.5):
            pred.append(-1)
    return pred

# f(w) = l(w) + theta*||w||-1
def obj(w, X, y, theta):
    z = np.dot(w.T, X)
    val = -np.multiply(y, z)
    J = np.sum(np.log(1+np.exp(val)))
    return J + theta*np.sum(np.abs(w))

def grad(w, X, y, theta):
    z = np.dot(w.T, X)
    val = -np.multiply(y, z)
    num = -np.multiply(y, np.exp(val))
    den = 1+np.exp(val)
    f = num/den
    gradJ = np.dot(X, f.T)
    return gradJ

def proxMap(L, w):
    return np.sign(w)*np.maximum(np.zeros(np.shape(w)), np.abs(w)-theta/L)

y = ytrain
X = xtrain
theta = 0.05
tol = 0.001
# Momentum
tOld = 1
tNew = 1
beta = 0
# Initialize w to be zero
wOld = np.zeros((58, 1))
w_array = np.array(wOld)
wBar = wOld
# L is Lipschitz constant
L = 0.5*(np.linalg.norm(xtrain)**2)*0.1
fx = obj(wOld, X, y, theta)  # Function value
gradwBar = grad(wBar, X, y, theta)  # Gradient at wBar
w = proxMap(L, wBar - gradwBar/L)  # Proximal mapping of g
gradw = grad(w, X, y, theta)  # Gradient at next w
err = np.sum((w - proxMap(np.ones((58, 1)), w - gradw))**2)**0.5
# Change in function value from w to next w
f_change = np.linalg.norm(obj(w, X, y, theta) - obj(wOld, X, y, theta))
err3 = np.linalg.norm(gradw)  # 2-norm of gradient
fx_lst = []
count = 0
maxit = 50
while count < maxit:  # and f_change > tol
    # Update momentum
    beta = (tOld-1)/tNew
    tOld = tNew
    tNew = 0.5 + 0.5*np.sqrt(1+4*(tOld**2.))
    # Iterate for next w
    wBar = w+(w-wOld)*theta
    gradwBar = grad(wBar, X, y, theta)
    wOld = w
    w = proxMap(L, wBar - gradwBar/L)
    gradw = grad(w, X, y, theta)
    fx = obj(w, X, y, theta)
    fx_lst.append(fx)
    err = np.sum((w - proxMap(np.ones((58, 1)), w - gradw))**2)**0.5
    f_change = np.linalg.norm(obj(w, X, y, theta) - obj(wOld, X, y, theta))
    err3 = np.linalg.norm(gradw)
    count += 1
    ypred = prediction(w, xtest)
    score = accuracy_score(ytest[0], ypred)*100
    print("Count", count, "f_diff", f_change, "fx", fx, "acc", score)

rang = range(0, count)


# In[305]:


plt.plot(rang, fx_lst)
plt.show()


# In[ ]:
