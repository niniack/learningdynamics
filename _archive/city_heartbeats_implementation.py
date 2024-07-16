import math
import os
import pickle
import random

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from numpy import linalg as LA
from tqdm import tqdm


# implementation DMD function
def H_DMD(X, delay):
    H = np.zeros((delay * X.shape[0], X.shape[1] - delay + 1))
    for k in range(delay):
        H[X.shape[0] * k : X.shape[0] * (k + 1), :] = X[
            :, k : (k + X.shape[1] - delay + 1)
        ]
    X1 = H[:, :-1]
    X2 = H[:, 1:]
    u, s, vh = LA.svd(X1, full_matrices=False)
    s[s > 0] = 1 / s[s > 0]
    # s = 1/s
    K = np.matrix.getH(u) @ X2 @ np.matrix.getH(vh) @ np.diag(s)
    Eigval, y = LA.eig(K)
    Eigvec = u @ y
    b0 = LA.pinv(Eigvec) @ X1[:, 0]
    return Eigval, Eigvec, b0, X1, X2, H, K


def DMD(X, *args):
    if len(args) == 0:
        X1 = X[:, :-1]
        X2 = X[:, 1:]
    elif len(args) == 1:
        X1 = X
        X2 = args[0]
    else:
        raise ValueError("Maximum 2 arguments are allowed")
    u, s, vh = LA.svd(X1, full_matrices=False)
    s[s > 0] = 1 / s[s > 0]
    # s = 1/s
    K = np.matrix.getH(u) @ X2 @ np.matrix.getH(vh) @ np.diag(s)
    Eigval, y = LA.eig(K)
    Eigvec = u @ y
    b0 = LA.pinv(Eigvec) @ X1[:, 0]
    return Eigval, Eigvec, b0, K


def Reverse_H(H, X_shape0, X_shape1):
    X_est = np.zeros((X_shape0, X_shape1))
    repeat_num = np.zeros((1, X_shape1))
    delay = int(H.shape[0] / X_shape0)
    for k in range(delay):
        X_est[:, k : (k + X_shape1 - delay + 1)] = (
            X_est[:, k : (k + X_shape1 - delay + 1)]
            + H[X_shape0 * k : X_shape0 * (k + 1), :]
        )
        repeat_num[:, k : (k + X_shape1 - delay + 1)] = (
            repeat_num[:, k : (k + X_shape1 - delay + 1)] + 1
        )
    for k in range(X_shape1):
        X_est[:, k] = X_est[:, k] / repeat_num[0, k]
    return X_est, repeat_num


def Reverse_H_new(vectors, n):
    # Infer d and m from the input matrix dimensions and vector length
    d = vectors.shape[0] // n
    m = vectors.shape[1] + d - 1

    hankel_matrix = np.zeros((n * d, vectors.shape[1]))
    mean_vectors = np.zeros((n, m))

    # Fill the Hankel matrix and collect the mean vectors
    for i in range(m):
        anti_diag_vectors = []

        # Collect vectors for the current anti-diagonal
        for j in range(max(0, i - d + 1), min(i + 1, vectors.shape[1])):
            vector_idx = i - j
            vector = vectors[vector_idx * n : (vector_idx + 1) * n, j]
            anti_diag_vectors.append(vector)

        # Calculate the mean vector for the current anti-diagonal
        if anti_diag_vectors:
            mean_vector = np.mean(anti_diag_vectors, axis=0)
            mean_vectors[:, i] = mean_vector

            # Assign the mean vector to the appropriate cells in the Hankel matrix
            for j in range(max(0, i - d + 1), min(i + 1, vectors.shape[1])):
                vector_idx = i - j
                hankel_matrix[vector_idx * n : (vector_idx + 1) * n, j] = mean_vector

    return hankel_matrix, mean_vectors


# implementation constrained DMD function
def vandermonde_matrix(lambdas, m):
    """
    Generates a Vandermonde matrix where the last column is raised to the power of (m-1).

    Parameters:
    lambdas (list): A list of lambda values [λ1, λ2, ..., λm].
    m (int): The scalar value for the power of the last column.
    """
    return np.vander(lambdas, m, increasing=True)


def generate_matrix_C(c):
    """
    Generates a matrix as shown in the picture given an array c as the last column.

    Parameters:
    c (numpy array): An array of size (m,) representing the last column of the matrix.

    Returns:
    numpy.ndarray: Resulting matrix with shape (m, m).
    """
    m = len(c)
    C = np.zeros((m, m))
    np.fill_diagonal(C[1:], 1)
    C[:, -1] = c
    return C


## Experiment example

### Run Hankel on 3-Day period data for training

# austria_graz

# Select 30 random rows from the matrix
austria_graz_matrix = austria_graz_matrix.dropna()

random.seed(10)
random_rows = random.sample(range(austria_graz_matrix.shape[0]), 30)
selected_matrix = austria_graz_matrix.iloc[random_rows, :]

# de-mean process
sub_selected_mat = selected_matrix.iloc[:, : int(24 * 60 / 5) * 4]
row_mean = sub_selected_mat.mean(axis=1)
normalized_matrix = sub_selected_mat.sub(row_mean, axis=0)

X = normalized_matrix.iloc[:, : int(24 * 60 / 5) * 3]
X_future = normalized_matrix.iloc[:, int(24 * 60 / 5) * 3 : int(24 * 60 / 5) * 4]

K_true = X
K_future = X_future

# HDMD results
delay = 300

temp = X
Eigval, Eigvec, bo, X1, X2, H, K = H_DMD(K_true.to_numpy(), delay)

H_koopman = np.zeros((H.shape))
for i in range(H.shape[1]):
    H_koopman[:, i] = np.real(Eigvec @ np.diag(Eigval**i) @ bo)

_, KMD_estimates = Reverse_H_new(H_koopman, X.shape[0])
KMD_estimates = KMD_estimates[:, : X.shape[1]]


# predict the future
H_future = np.zeros((H.shape[0], K_future.shape[1]))
for j in range(H_future.shape[1]):
    i = j + temp.shape[1]
    temp_vec = np.real(Eigvec @ np.diag(np.exp(np.log(Eigval) * i)) @ bo)
    H_future[:, j] = temp_vec

_, KMD_predicts = Reverse_H_new(H_future, K_future.shape[0])
KMD_predicts = KMD_predicts[:, : K_future.shape[1]]

# Store the eigen-triples to local for future usage
with open("objs_graz_H300.pkl", "wb") as f:
    pickle.dump([Eigvec, Eigval, bo], f)

# plotting
vmin = K_true.min().min()
vmax = K_true.max().max()

print(LA.norm(KMD_estimates - K_true, "fro"))
print(LA.norm(KMD_estimates - K_true, "fro") / LA.norm(temp, "fro"))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(wspace=0.1, top=0.85)

# Plotting the first subplot
plt.subplot(121)
plt.imshow(K_true, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
plt.xticks(
    np.arange(0, int(24 * 60 / 5) * 3, step=60 / 5 * 6),
    np.arange(0, 24 * 3, step=6),
    rotation=45,
)
plt.yticks(
    np.arange(0, len(K_true), step=5), K_true.index[::5]
)  # Display every 5th name
plt.xlabel("Time (hr)", fontsize=12)
plt.ylabel("Detector Index", fontsize=12)
plt.title("Groundtruth 2016-04-04~06", fontsize=12)

# Plotting the second subplot
plt.subplot(122)
im = plt.imshow(KMD_estimates, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
plt.xticks(
    np.arange(0, int(24 * 60 / 5) * 3, step=60 / 5 * 6),
    np.arange(0, 24 * 3, step=6),
    rotation=45,
)
plt.yticks([])
plt.xlabel("Time (hr)", fontsize=12)
plt.title("Estimated 2016-04-04~06", fontsize=12)

# Adding colorbar for the second subplot
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Define position of colorbar
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Mean-reduced Flow (veh/h-lane)")

# plt.suptitle('Austria Graz - HDMD reconstruction', fontsize=12)
plt.savefig("Graz-train.pdf", bbox_inches="tight")
plt.show()


# plotting
vmin = K_future.min().min()
vmax = K_future.max().max()

print(LA.norm(KMD_predicts - K_future, "fro"))
print(LA.norm(KMD_predicts - K_future, "fro") / LA.norm(K_future, "fro"))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(wspace=0.1, top=0.85)

# Plotting the first subplot
plt.subplot(121)
plt.imshow(K_future, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
plt.xticks(
    np.arange(0, int(24 * 60 / 5) * 1, step=60 / 5 * 3),
    np.arange(0, 24 * 1, step=3),
    rotation=45,
)
plt.yticks(
    np.arange(0, len(K_future), step=5), K_future.index[::5]
)  # Display every 5th name
plt.xlabel("Time (hr)", fontsize=12)
plt.ylabel("Detector Index", fontsize=12)
plt.title("Groundtruth 2016-04-07", fontsize=12)

# Plotting the second subplot
plt.subplot(122)
im = plt.imshow(KMD_predicts, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
plt.xticks(
    np.arange(0, int(24 * 60 / 5) * 1, step=60 / 5 * 3),
    np.arange(0, 24 * 1, step=3),
    rotation=45,
)
plt.yticks([])
plt.xlabel("Time (hr)", fontsize=12)
plt.title("Prediction 2016-04-07", fontsize=12)

# Adding colorbar for the second subplot
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Define position of colorbar
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Mean-reduced Flow (veh/h-lane)")

# plt.suptitle('Austria Graz - HDMD prediction', fontsize=12)
plt.savefig("Graz-test.pdf", bbox_inches="tight")
plt.show()
