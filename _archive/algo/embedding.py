import einops
import torch


def make_hankel_matrix(input_matrix, h_delay=10):
    hankel_matrix = input_matrix.unfold(
        dimension=1, size=(input_matrix.shape[-1] - h_delay + 1), step=1
    )

    hankel_matrix = einops.rearrange(hankel_matrix, "w o t -> (o w) t")
    return hankel_matrix


def hankel(X, delays=10, lag=1):
    """
    Given a data matrix X as a 1-D or 2-D torch.Tensor, uses the `delays`
    and `lag` attributes to return the data as a 2-D Hankel matrix.

    :param X: (m,) or (n, m) array of data.
    :type X: torch.ndarray
    :return: Hankel matrix of data.
    :rtype: torch.Tensor
    """
    if not isinstance(X, torch.Tensor) or X.ndim > 2:
        raise ValueError("Data must be a 1-D or 2-D torch tensor.")

    if X.ndim == 1:
        X = X[None]

    n, m = X.shape

    # Check that the input data contains enough observations.
    m_min = lag * (delays - 1) + 1
    if m < m_min:
        raise ValueError(
            "Not enough snapshots provided for "
            f"{delays} delays and lag {lag}. "
            f"Please provide at least {m_min} snapshots."
        )

    Hm = m - ((delays - 1) * lag)
    H = torch.empty((n * delays, Hm))
    for i in range(delays):
        H[i * n : (i + 1) * n] = X[:, i * lag : i * lag + Hm]

    return H


def dehankel(H, n, lag=1):
    """
    Given a Hankel matrix H and the number of rows per time series (n),
    deconstructs it into the original time series data.

    :param H: (n * delays, m - (delays - 1) * lag) Hankel matrix.
    :type H: torch.Tensor
    :param n: Number of rows per time series in the Hankel matrix.
    :type n: int
    :param lag: Lag between samples (default 1).
    :type lag: int
    :return: Reconstructed time series data.
    :rtype: torch.Tensor
    """

    if not isinstance(H, torch.Tensor) or H.ndim != 2:
        raise ValueError("Hankel matrix must be a 2-D torch tensor.")

    delays, Hm = H.shape

    if delays % n != 0:
        raise ValueError("Number of rows in Hankel matrix must be divisible by n.")

    m = Hm + (delays // n - 1) * lag  # Calculate the original length

    X = torch.zeros((n, m))

    for i in range(n):
        for j in range(delays // n):
            X[i, j * lag : j * lag + Hm] = H[j * n + i]

    return X
