import numpy as np
from scipy.signal.windows import gaussian, exponential, tukey, blackmanharris


def hanning2D(n):
    h = np.hanning(n)
    return np.sqrt(np.outer(h, h))

def gaussian2d(n, std):
    g = gaussian(n, std)
    return np.sqrt(np.outer(g, g))

def exponential2d(n, tau):
    e = exponential(n, tau=tau)
    return np.sqrt(np.outer(e, e))

def tukey2d(n, alpha):
    t = tukey(n, alpha=alpha)
    return np.sqrt(np.outer(t, t))

def blackmanharris2d(n):
    b = blackmanharris(n)
    return np.sqrt(np.outer(b, b))

def highpass2d(size):
    rows = np.cos(np.pi * np.matrix([-0.5 + x/(size[0] - 1) for x in range(size[0])]))
    cols = np.cos(np.pi * np.matrix([-0.5 + x/(size[1] - 1) for x in range(size[1])]))
    X = np.outer(rows, cols)
    return (1.0 - X) * (2.0 - X)

def laplacian2d(size):
    h = np.zeros_like(size, dtype=np.float32)
    constFactor = (-4) * np.pi * np.pi
    for i in range(size[0]):
        iFactor = (i - size[0]/2)**2
        for j in range(size[1]):
            h[i, j] = constFactor * ( iFactor + (j - size[1]/2)**2 )
