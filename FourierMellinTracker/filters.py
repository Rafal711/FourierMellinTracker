import numpy as np

def hanning2D(n):
    h = np.hanning(n)
    return np.sqrt(np.outer(h, h))

def highpassFilter(size):
    rows = np.cos(np.pi * np.matrix([-0.5 + x/(size[0] - 1) for x in
    range(size[0])]))
    cols = np.cos(np.pi * np.matrix([-0.5 + x/(size[1] - 1) for x in
    range(size[1])]))
    X = np.outer(rows, cols)
    return (1.0 - X) * (2.0 - X)