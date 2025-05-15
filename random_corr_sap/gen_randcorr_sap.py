import numpy as np


def randcorr(p):
    # Check inputs
    if not isinstance(p, int) or p < 2:
        raise ValueError("p must be a scalar integer greater than 1.")

    # Step 1 - generate angles theta from PDF (sin(theta))^k, k >= 1, 0 < theta < pi
    e = np.ones((p, ))
    theta = np.zeros((p, p))
    for j in range(p - 1):
        #print('j=%d' % j)
        theta[(j + 1):p, j] = randcorr_sample_sink((p - j) * e[(j + 1):p])

    # Step 2 - construct lower triangular Cholesky factor
    L = np.ones((p, p))
    for i in range(1, p):
        L[i, 1:i+1] = np.cumprod(np.sin(theta[i, 0:i]))

    R = np.cos(theta)
    R[np.triu_indices(p, 1)] = 0
    L = L * R

    # Form correlation matrix
    C = np.dot(L, L.T)

    return C


def randcorr_sample_sink(k):
    N = k.shape[0]
    logconst = 2 * np.log(np.pi / 2)
    
    # Sampling loop - vectorized
    x = np.zeros(N)
    accept = np.full(N, False)
    
    while not np.all(accept):
        # index of samples that need to be accepted
        ix = ~accept
        T = np.sum(ix)
        
        # Beta(k+1, k+1) rng
        x[ix] = np.pi * np.random.beta(k[ix] + 1, k[ix] + 1, size=T)
        
        # Check acceptance
        accept[ix] = np.log(np.random.uniform(size=T)) / k[ix] < logconst + np.log(np.sin(x[ix])) - np.log(x[ix]) - np.log(np.pi - x[ix])
         
    return x
