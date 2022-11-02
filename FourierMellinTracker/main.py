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
pattern = None


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

def setPatternArea(frame):
    global pattern
    leftSide = mouseXY1[0] - squareHalfSide
    rightSide = mouseXY1[0] + squareHalfSide
    upperSide = mouseXY1[1] - squareHalfSide
    bottomSide = mouseXY1[1] + squareHalfSide
    pattern = frame[upperSide:bottomSide, leftSide:rightSide]

def handleMouseCallback():
    cv2.setMouseCallback('frame', getMousePosition)

def fpsToDelayTime(fps):
    return int(1000/fps)

def startVideoProcessing():
    isObjectVisible = False

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

def startVideoObjectTracking():
    moviePath = r"D:\movies\domek.mp4"   # 0 dla kamery
    video = cv2.VideoCapture(0)
    isObjectVisible = True

    if not video.isOpened():
        print("Cannot open video/camera")
        exit()

    current_frame = 0
    frames_per_second = 5
    frame_rate = video.get(cv2.CAP_PROP_FPS)  # video frame rate
    if frames_per_second > frame_rate or frames_per_second == -1:
        frames_per_second = frame_rate

    global mouseXY1
    objTracker = FourierMellinTracker(filters.hanning2D, filters.highpass2d)
    delayTime = fpsToDelayTime(100)

    while True:
        frameIsReady, frame = video.read()
        if not frameIsReady:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # if isObjectVisible:
        drawTrackingBox(frame)
        drawPointForSelectedObject(frame)
        handleMouseCallback()


        if current_frame % (math.floor(frame_rate / frames_per_second)) == 0:
            cv2.imshow('frame', frame)

            if state == State.HalfLengthOfSquareSelected:
                setPatternArea(grayFrame)
                if pattern is not None:
                    objTracker.objectTracking(pattern, grayFrame, mouseXY1)
                    mouseXY1 = objTracker.positionGlobal
                    isObjectVisible = objTracker.objectIsVisible

        if cv2.waitKey(1) == ord('q'):
            break

        current_frame += 1
        #cv2.waitKey(delayTime)

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

    print(searchedImgs)

    # patternSection = cv2.imread(imgsPath + 'wzor.pgm', cv2.IMREAD_GRAYSCALE)
    # searchedSection = cv2.imread(imgsPath + searchedImgs[7], cv2.IMREAD_GRAYSCALE)

    # patternSection = cv2.imread(imgsPath + 'domek_r0_64.pgm', cv2.IMREAD_GRAYSCALE)
    # searchedSection = cv2.imread(imgsPath + "domek_r30.pgm", cv2.IMREAD_GRAYSCALE)

    patternSection = cv2.imread("testImgs/patternSmall.png", cv2.IMREAD_GRAYSCALE)
    searchedSection = cv2.imread("testImgs/search1.png", cv2.IMREAD_GRAYSCALE)

    obj1 = FourierMellinTracker(filters.hanning2D, filters.highpass2d)
    obj1.objectTracking(patternSection, searchedSection, (None, None), True)


if __name__ == '__main__':
    # startVideoProcessing()
    # testing_1()
    startVideoObjectTracking()