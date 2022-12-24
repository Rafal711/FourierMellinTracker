import cv2
import numpy as np
import pandas as pd
import math
import filters
import os
from enum import Enum

import imagePlot
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
            squareHalfSide = int(round(math.dist(mouseXY1, mouseXY2)))
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

def drawTrackingBox(frame, objectIsVisible=True):
    if state != State.HalfLengthOfSquareSelected:
        return

    pLeftUpper = (mouseXY1[0] - squareHalfSide, mouseXY1[1] + squareHalfSide)
    pRightBottom = (mouseXY1[0] + squareHalfSide, mouseXY1[1] - squareHalfSide)
    pRightUpper = (mouseXY1[0] + squareHalfSide, mouseXY1[1] + squareHalfSide)
    pLeftBottom = (mouseXY1[0] - squareHalfSide, mouseXY1[1] - squareHalfSide)

    cv2.rectangle(frame, pLeftUpper, pRightBottom, (0, 0, 255), 1)
    if not objectIsVisible:
        cv2.line(frame, pLeftUpper, pRightBottom, (0, 0, 255), 1)
        cv2.line(frame, pRightUpper, pLeftBottom, (0, 0, 255), 1)

def setPatternArea(frame):
    leftSide = mouseXY1[0] - squareHalfSide
    rightSide = mouseXY1[0] + squareHalfSide
    upperSide = mouseXY1[1] - squareHalfSide
    bottomSide = mouseXY1[1] + squareHalfSide
    return frame[upperSide:bottomSide, leftSide:rightSide]

def handleMouseCallback():
    cv2.setMouseCallback('frame', getMousePosition)

def fpsToDelayTime(fps):
    return int(1000/fps)


def startVideoObjectTrackingFourierMellin():
    movieNames = ["car4", "car11", "david_indoor", "trellis"]
    moviePath = "testVideos/"
    movieFormat = ".avi"
    choosenMovie = moviePath + movieNames[0] + movieFormat

    video = cv2.VideoCapture(0)  # 0 dla kamery
    if not video.isOpened():
        print("Cannot open video/camera")
        exit()

    global mouseXY1, squareHalfSide
    current_frame = 0
    objectIsVisible = True
    frameRate = video.get(cv2.CAP_PROP_FPS)
    playVideo = True
    initializePattern = False
    if initializePattern:
        mouseXY1 = (111, 96)
        squareHalfSide = 52
        global state
        state = State.HalfLengthOfSquareSelected

    framesPerSecond = 5
    if framesPerSecond > frameRate or framesPerSecond == -1:
        framesPerSecond = frameRate
    # delayTime = fpsToDelayTime(100)
    ## dim = (width, height)
    # dim = (800, 600)

    frameWidth = int(video.get(3))
    frameHeight = int(video.get(4))
    size = (frameWidth, frameHeight)
    # fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    # recorder = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size)

    objTracker = FourierMellinTracker(filters.hanning2D, filters.highpass2d)

    while True:
        if playVideo:
            frameIsReady, frame = video.read()
            if not frameIsReady:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if True: #current_frame % (math.floor(frameRate / framesPerSecond)) == 0:
            if (state == State.HalfLengthOfSquareSelected or initializePattern) and playVideo:
                objTracker.objectTracking(setPatternArea(grayFrame), grayFrame, mouseXY1)
                mouseXY1 = objTracker.positionGlobal
                objectIsVisible = objTracker.objectIsVisible
                squareHalfSide = objTracker.pattern.shape[0] // 2
            drawTrackingBox(frame, objectIsVisible)
            drawPointForSelectedObject(frame)
            handleMouseCallback()
            cv2.imshow('frame', frame)

        if state == State.PointsNotSelected:
            objTracker.pattern = None

        if cv2.waitKey(1) == ord('q'):
            break

        if cv2.waitKey(5) == ord('p'):
            playVideo = not playVideo
            if playVideo:
                print("movie play")
            else:
                print("movie pauze")

        # recorder.write(frame)

        current_frame += 1
        # cv2.waitKey(delayTime)

    video.release()
    # recorder.release()
    cv2.destroyAllWindows()


# dynamika - polo, ballInd1
# obrót wokól własnej osi - bag2, cartman, robotCorridor
# przezroczysty obiekt - 4sphere
# powolny obrot - boxroom1
# ruch kamery, obrot itp. - 4caraftertree, lantern
# sam ruch kamery - Fist
def startVideoObjectTrackingFourierMellin2():
    rest = ['polo', 'ballInd1', 'bag2', 'cartman', 'robotCorridor', '4sphere', 'boxroom1', '4caraftertree', 'lantern', 'Fish']
    best = ['mug', '4bus6', 'boxroom1', 'Sylvester']
    delete = []
    movieNames = ['boxRoom31']

    choosenWideoId = 0 # movieNames.index("mug")

    # moviesNamesList = os.listdir("testVideosFinal"); moviesNamesList.remove("skrypty"); moviesNamesList.remove("calculateGT.py")
    # print(moviesNamesList)

    dataPath = "testVideosFinal/" + movieNames[choosenWideoId] + "/"
    dataGT = pd.read_csv(dataPath + "groundtruth.csv", delimiter = ",", index_col=0)

    startCoord = (int(dataGT["Xc"][0]), int(dataGT["Yc"][0]))
    startWindow = int(dataGT["R"][0])

    global mouseXY1, squareHalfSide
    current_frame = 0
    counter = 0
    objectIsVisible = True
    playVideo = True
    initializePattern = True
    if initializePattern:
        mouseXY1 = startCoord
        squareHalfSide = startWindow
        global state
        state = State.HalfLengthOfSquareSelected

    objTracker = FourierMellinTracker(filters.hanning2D, filters.highpass2d)

    wideoPath = "testVideosFinal/" + movieNames[choosenWideoId] + "/wideo"
    wideoFramesNames = os.listdir(wideoPath)

    firstFrame = cv2.imread(wideoPath + "/" + wideoFramesNames[0])
    frameWidth = int(firstFrame.shape[1])
    frameHeight = int(firstFrame.shape[0])
    size = (frameWidth, frameHeight)
    # recorder = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size)

    while True:
        if playVideo and counter < len(wideoFramesNames):
            frame = cv2.imread(wideoPath + "/" + wideoFramesNames[counter]) #cv2.IMREAD_GRAYSCALE
            counter += 1
        else:
            break

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if True: #current_frame % (math.floor(frameRate / framesPerSecond)) == 0:
            if (state == State.HalfLengthOfSquareSelected or initializePattern) and playVideo:
                objTracker.objectTracking(setPatternArea(grayFrame), grayFrame, mouseXY1)
                mouseXY1 = objTracker.positionGlobal
                objectIsVisible = objTracker.objectIsVisible
                squareHalfSide = objTracker.pattern.shape[0] // 2
            drawTrackingBox(frame, objectIsVisible)
            drawPointForSelectedObject(frame)
            # cv2.circle(frame, (235, 68), radius=1, color=(0, 0, 255), thickness=-1)
            handleMouseCallback()
            cv2.imshow('frame', frame)

        if state == State.PointsNotSelected:
            objTracker.pattern = None

        if cv2.waitKey(1) == ord('q'):
            break

        if cv2.waitKey(5) == ord('p'):
            playVideo = not playVideo
            if playVideo:
                print("movie play")
            else:
                print("movie pauze")

        # recorder.write(frame)

        current_frame += 1
        # cv2.waitKey(delayTime)
    cv2.destroyAllWindows()
    # recorder.release()


def checkPatternCoordinates():
    movieNames = ["car4", "car11", "david_indoor", "trellis"]
    moviePath = "testVideos/"
    movieFormat = ".avi"
    choosenMovie = moviePath + movieNames[0] + movieFormat

    video = cv2.VideoCapture(choosenMovie)  # 0 dla kamery
    if not video.isOpened():
        print("Cannot open video/camera")
        exit()

    global mouseXY1, squareHalfSide

    playVideo = True
    objectIsVisible = True

    frameWidth = int(video.get(3))
    frameHeight = int(video.get(4))
    size = (frameWidth, frameHeight)

    while True:
        if playVideo:
            frameIsReady, frame = video.read()
            if not frameIsReady:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            playVideo = False
            # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        drawTrackingBox(frame, objectIsVisible)
        drawPointForSelectedObject(frame)
        handleMouseCallback()
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

        if cv2.waitKey(5) == ord('p'):
            playVideo = not playVideo
            if playVideo:
                print("movie play")
            else:
                print("movie pauze")
        print(mouseXY1, squareHalfSide)
    video.release()
    cv2.destroyAllWindows()


def imageTesting():
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
    # imageTesting()
    # startVideoObjectTrackingFourierMellin()
    startVideoObjectTrackingFourierMellin2()
    # startVideoObjectTrackingMeanShift()
    # checkPatternCoordinates()