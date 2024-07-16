import einops
import numpy as np
import torch
from numpy import linalg as LA


class HankelDMD:
    def __init__(self, input_matrix, h_delay):
        self.input_matrix = input_matrix
        self.h_delay = h_delay

    def make_hankel_matrix(self):
        hankel_matrix = self.input_matrix.unfold(
            dimension=1,
            size=(self.input_matrix.shape[-1] - self.h_delay + 1),
            step=1,
        )

        hankel_matrix = einops.rearrange(hankel_matrix, "w o t -> (o w) t")
        return hankel_matrix

    def execute_dmd(self):
        # Build Hankel matrix
        hankel_matrix = self.make_hankel_matrix()

        # Time-shifted Hankel Matrices
        hankel_x_0 = hankel_matrix[:, :-1]  # All columns except the last
        hankel_x_1 = hankel_matrix[:, 1:]  # All columns except the first

        # SVD on X0
        U, S, Vh = torch.linalg.svd(hankel_x_0, full_matrices=False)

        # Invert singular values
        inv_S = torch.linalg.inv(torch.diag(S))

        # Linear operator computation
        operator_K = U.H @ hankel_x_1 @ Vh.H @ inv_S

        # Obtain eigenvalues and eigenvectors
        eigenvalues, Y = torch.linalg.eig(operator_K)
        eigenvectors = U.type(torch.complex64) @ Y
        b0 = torch.linalg.pinv(eigenvectors) @ hankel_x_1[:, 0].type(torch.complex64)

        return eigenvalues
        # return (
        #     eigenvalues,
        #     eigenvectors,
        #     b0,
        #     hankel_x_0,
        #     hankel_x_1,
        #     hankel_matrix,
        #     operator_K,
        # )


def H_DMD(X, delay):
    H = torch.zeros((delay * X.shape[0], X.shape[1] - delay + 1))
    H = H.numpy()
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

    return Eigval
    # return Eigval, Eigvec, b0, X1, X2, H, K
