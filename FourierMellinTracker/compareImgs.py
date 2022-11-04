import numpy as np
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity


def mse(imageA, imageB):
    error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    error /= float(imageA.shape[0] * imageA.shape[1])
    return error

def ssim(imageA, imageB):
    return structural_similarity(imageA.astype("float"), imageB.astype("float"))

def averageHash(imageA, imageB):
    hashA = imagehash.average_hash(Image.fromarray(imageA))
    hashB = imagehash.average_hash(Image.fromarray(imageB))
    cutoff = 5

    return (hashA - hashB) < cutoff
