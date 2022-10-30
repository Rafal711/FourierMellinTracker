import cv2
import numpy as np
import math
import filters
import os
from enum import Enum
from fourierMellinTracker import FourierMellinTracker


class State(Enum):
    PointsNotSelected = 0
    MiddleOfObjectSelected = 1
    HalfLengthOfSquareSelected = 2


state = State.PointsNotSelected
squareHalfSide = 0
mouseXY1 = (None, None)
mouseXY2 = (None, None)


def getMousePosition(event, x, y, flags, params):
    global state, mouseXY1, mouseXY2, squareHalfSide
    if event == cv2.EVENT_LBUTTONUP:
        if state == State.PointsNotSelected:
            mouseXY1 = x, y
            state = State.MiddleOfObjectSelected
        elif state == State.MiddleOfObjectSelected:
            mouseXY2 = x, y
            squareHalfSide = round(math.dist(mouseXY1, mouseXY2) / 2) * 2
            state = State.HalfLengthOfSquareSelected
    elif event == cv2.EVENT_RBUTTONUP:
        mouseXY1 = None, None
        mouseXY2 = None, None
        state = State.PointsNotSelected

def isMousePointDefined(point):
    return all([coord is not None for coord in point])

def drawPointForSelectedObject(frame):
    if isMousePointDefined(mouseXY1):
        cv2.circle(frame, mouseXY1, radius=1, color=(0, 0, 255), thickness=-1)

def drawTrackingBox(frame):
    if state != State.HalfLengthOfSquareSelected:
        return
    pLeftUpper = (mouseXY1[0] - squareHalfSide, mouseXY1[1] + squareHalfSide)
    pRightBottom = (mouseXY1[0] + squareHalfSide, mouseXY1[1] - squareHalfSide)
    cv2.rectangle(frame, pLeftUpper, pRightBottom, (0, 0, 255), 1)


def handleMouseCallback():
    cv2.setMouseCallback('frame', getMousePosition)

def startVideoProcessing():
    startFourierMellinTracking = False

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Cannot open video/camera")
        exit()

    while True:
        frameIsReady, frame = video.read()
        if not frameIsReady:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        drawTrackingBox(frame)
        drawPointForSelectedObject(frame)
        cv2.imshow('frame', frame)
        handleMouseCallback()

        if cv2.waitKey(1) == ord('q'):
            break

        print(mouseXY1)
        print(mouseXY2)
        print(state)

    video.release()
    cv2.destroyAllWindows()


def testing_1():
    path = "obrazy_Mellin"
    imgsPath = "obrazy_Mellin/"
    dir_list = os.listdir(path)
    temp_pos = (None, None)

    searchedImgs = dir_list[:-1]
    searchedImgs.remove('domek_r0_64.pgm')
    searchedImgs.remove('wzor.pgm')

    patternSection = cv2.imread(imgsPath + 'wzor.pgm', cv2.IMREAD_GRAYSCALE)
    searchedSection = cv2.imread(imgsPath + searchedImgs[23], cv2.IMREAD_GRAYSCALE)

    # patternSection = cv2.imread(imgsPath + 'domek_r0_64.pgm', cv2.IMREAD_GRAYSCALE)
    # searchedSection = cv2.imread(imgsPath + "domek_r30.pgm", cv2.IMREAD_GRAYSCALE)

    obj1 = FourierMellinTracker(temp_pos, filters.hanning2D, filters.highpass2d)
    obj1.objectTracking(patternSection, searchedSection)


if __name__ == '__main__':
    startVideoProcessing()
    # testing_1()
