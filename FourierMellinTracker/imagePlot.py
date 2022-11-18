import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import copy


def plotImage(img, title=""):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
    ax.set_title(title)
    plt.show()

def plotImages1x2(pattern, searchedImg, orgSize, position, visibilityFlag=True, frameNr=0, similarity=0):
    squareX = position[0] - orgSize[0] // 2
    squareY = position[1] - orgSize[1] // 2
    box1 = patches.Rectangle((squareX, squareY), orgSize[0], orgSize[1],
                             linewidth=1, edgecolor='r', facecolor='none')
    box2 = copy.copy(box1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

    ax1.imshow(pattern, cmap='gray', vmin=0, vmax=255)
    ax1.plot(position[0], position[1], "ro")
    ax1.set_title("pattern scaled and rotated")
    if visibilityFlag:
        ax1.add_patch(box1)
    ax1.set_xlim(0, pattern.shape[0])
    ax1.set_ylim(pattern.shape[1], 0)

    ax2.imshow(searchedImg, cmap='gray', vmin=0, vmax=255)
    ax2.plot(position[0], position[1], "ro")
    ax2.set_title(f"searched image {similarity}")
    ax2.set_xlim(0, pattern.shape[0])
    ax2.set_ylim(pattern.shape[1], 0)
    if visibilityFlag:
        ax2.add_patch(box2)
    plt.savefig(f'testVideoResult/foo{frameNr}.png')
    # plt.show()

def plot3dImage(img):
    ax = plt.axes(projection='3d')
    xx, yy = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    X, Y = xx.ravel(), yy.ravel()
    Z = np.zeros_like(img).ravel()
    dx = .25 * np.ones(img.size)
    dy = .25 * np.ones(img.size)
    dz = img.ravel()
    ax.bar3d(X, Y, Z, dx, dy, dz, color = 'w')
    plt.show()
