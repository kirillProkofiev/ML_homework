import numpy as np
from scipy import linalg
import numpy.linalg as LA
import matplotlib.pyplot as plt

svd = linalg.svd
eig = LA.eig
array = np.array

def pca(X: np.ndarray):
    N = X.shape[0]
    x_mean = X.mean(axis=0)
    print('x_mean: ', x_mean)
    X_centered = X - x_mean
    print('X_centered: ', X_centered)
    print(X_centered.T)
    C = (X_centered.T).dot(X_centered)
    print('C: ', C)
    C_norm = (1/(N-1))*C
    alpha, v = eig(C)
    print('alpha: ', alpha, 'v:', v)
    disp = (1/(N-1))*alpha
    print('disp: ', disp)
    part_of_disp = []
    for alp in alpha:
        part_of_disp.append(alp/sum(alpha))
    print('part_of_disp:', part_of_disp)
    plt.scatter(X_centered[:,0], X_centered[:,1])
    z = np.arange(-5,5,0.1)
    v1 = v[0,0]*z + v[1,0]
    v2 = v[0,1]*z + v[1,1]
    plt.plot(z, v1)
    plt.plot(z, v2)
    plt.show()

X = array([[4,0,-2,2],[3,1,-3,-1]]).T
pca(X)
