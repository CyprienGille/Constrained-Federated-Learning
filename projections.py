import numpy as np
import torch


def proj_l1inf_numpy(Y, c, tol=1e-5, direction="row"):
    """
    {X : sum_n max_m |X(n,m)| <= c}
    for some given c>0

        Author: Laurent Condat
        Version: 1.0, Sept. 1, 2017

    This algorithm is new, to the author's knowledge. It is based
    on the same ideas as for projection onto the l1 ball, see
    L. Condat, "Fast projection onto the simplex and the l1 ball",
    Mathematical Programming, vol. 158, no. 1, pp. 575-585, 2016.

    The algorithm is exact and terminates in finite time*. Its
    average complexity, for Y of size N x M, is O(NM.log(M)).
    Its worst case complexity, never found in practice, is
    O(NM.log(M) + N^2.M).

    Note : This is a numpy transcription of the original MATLAB code
    *Due to floating point errors, the actual implementation of the algorithm
    uses a tolerance parameter to guarantee halting of the program
    """
    added_dimension = False

    if Y.ndim == 1:
        Y = np.expand_dims(Y, axis=0)
        added_dimension = True

    if direction == "col":
        Y = np.transpose(Y)

    X = np.flip(np.sort(np.abs(Y), axis=1), axis=1)
    v = np.sum(X[:, 0])
    if v <= c:
        return Y
    N, M = Y.shape
    S = np.cumsum(X, axis=1)
    idx = np.ones((N, 1), dtype=int)
    theta = (v - c) / N
    mu = np.zeros((N, 1))
    active = np.ones((N, 1))
    theta_old = 0
    while np.abs(theta_old - theta) > tol:
        for n in range(N):
            if active[n]:
                j = idx[n]
                while (j < M) and ((S[n, j - 1] - theta) / j) < X[n, j]:
                    j += 1
                idx[n] = j
                mu[n] = S[n, j - 1] / j
                if j == M and (mu[n] - (theta / j)) <= 0:
                    active[n] = 0
                    mu[n] = 0
        theta_old = theta
        theta = (np.sum(mu) - c) / (np.sum(active / idx))
    X = np.minimum(np.abs(Y), (mu - theta / idx) * active)
    X = X * np.sign(Y)

    if direction == "col":
        X = np.transpose(X)

    if added_dimension:
        X = np.squeeze(X)
    return X


def proj_l1infball(w0, eta: float, device="cpu"):
    shape = w0.shape
    w = w0.numpy(force=True)
    res = proj_l1inf_numpy(w, eta, direction="col")
    Q = torch.as_tensor(res, dtype=torch.get_default_dtype(), device=device)
    return Q.reshape(shape)
