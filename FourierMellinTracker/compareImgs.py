import numpy as np
import filters
from skimage.metrics import structural_similarity


def mse(imageA, imageB):
    error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    error /= float(imageA.shape[0] * imageA.shape[1])
    return error

def ssim(imageA, imageB):
    return structural_similarity(imageA.astype("float"), imageB.astype("float"))
