import numpy as np
import cv2
import compareImgs
import filters
import imagePlot as iplt


class FourierMellinTracker:
    def __init__(self, edgeFilter = filters.hanning2D, highPassFilter = filters.highpass2d):
        self.edgeFilter = edgeFilter
        self.highPassFilter = highPassFilter
        self.positionMid = (None, None)
        self.positionGlobal = (None, None)
        self.positionGlobalShift = (0, 0)
        self.objectIsVisible = True
        self.searchedArea = None
        self.pattern = None

    def imageZeroPaddingLUC(self, image, targetSize):
        resultSize = (targetSize[0] - image.shape[0], targetSize[1] - image.shape[1])
        return cv2.copyMakeBorder(image, 0, resultSize[0], 0, resultSize[1], cv2.BORDER_CONSTANT, value=0)

    def imageZeroPaddingMid(self, image, targetSize):
        resultSize = ((targetSize[0] - image.shape[0]) // 2,
                      (targetSize[1] - image.shape[1]) // 2)
        return np.pad(image, resultSize, "constant", constant_values=0)

    def magnitude(self, imageFft):
        return np.abs(imageFft)

    def reduceEdgeEffects(self, image):
        return image * self.edgeFilter(image.shape[0])

    def logPolarTransform(self, image):
        center = (image.shape[0] // 2, image.shape[1] // 2)
        M = image.shape[0] / np.log(image.shape[0] // 2)
        imageLogPolar = cv2.logPolar(image, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        return imageLogPolar, M

    def phaseCorrelationMag(self, patternImgFft, searchedImgFft):
        imgsCrossCorrelationFft = np.multiply(searchedImgFft, np.conj(patternImgFft))
        imgsPhaseCorrelationFft = imgsCrossCorrelationFft / np.abs(imgsCrossCorrelationFft)
        return self.magnitude(np.fft.ifft2(imgsPhaseCorrelationFft))

    def calculateAnglesAndScale(self, imgsPhaseCorrelationMag, searchedImgLogPolarSize, M):
        angleCoord, scaleCoord = np.unravel_index(np.argmax(imgsPhaseCorrelationMag), imgsPhaseCorrelationMag.shape)
        scaleSize = searchedImgLogPolarSize[0]
        angleSize = searchedImgLogPolarSize[1]
        if scaleCoord > scaleSize // 2:
            scaleIndexFactor = scaleSize - scaleCoord  # increase
        else:
            scaleIndexFactor = - scaleCoord  # minimization
        A = (angleCoord * 360.0) / angleSize
        angles = (-A, 180 - A)
        scale = np.exp(scaleIndexFactor / M)
        return angles, scale

    def getRotatedScaledPatterns(self, patternImg, searchedImgSize, angles, scale):
        patternZP = self.imageZeroPaddingMid(patternImg, searchedImgSize)
        middleOfTranslation = (patternZP.shape[0] / 2 - 0.5, patternZP.shape[1] / 2 - 0.5)

        translationMatrix1 = cv2.getRotationMatrix2D(middleOfTranslation, angles[0], scale)
        imgRotatedScaled1 = cv2.warpAffine(patternZP, translationMatrix1, patternZP.shape)

        translationMatrix2 = cv2.getRotationMatrix2D(middleOfTranslation, angles[1], scale)
        imgRotatedScaled2 = cv2.warpAffine(patternZP, translationMatrix2, patternZP.shape)
        return imgRotatedScaled1, imgRotatedScaled2

    def bestTransformedPattern(self, img1, img2, searchedImgFft):
        img1Fft = np.fft.fft2(img1)
        img2Fft = np.fft.fft2(img2)

        img1PhaseCorrelation = self.phaseCorrelationMag(img1Fft, searchedImgFft)
        img2PhaseCorrelation = self.phaseCorrelationMag(img2Fft, searchedImgFft)

        if np.amax(img1PhaseCorrelation) > np.amax(img2PhaseCorrelation):
            betterPhaseCorr = img1PhaseCorrelation
            betterImg = img1
        else:
            betterPhaseCorr = img2PhaseCorrelation
            betterImg = img2

        dy, dx = np.unravel_index(np.argmax(betterPhaseCorr), betterPhaseCorr.shape)
        return betterImg, (dx, dy)

    def shiftImage(self, img, shift):
        shiftedImg = np.roll(img, shift[0], axis=1)  # horizontally
        return np.roll(shiftedImg, shift[1], axis=0)  # vertically

    def predictObjectPosition(self, searchedImgSize, shift):
        dxZP = (searchedImgSize[0] - self.pattern.shape[0]) // 2
        dyZP = (searchedImgSize[1] - self.pattern.shape[1]) // 2

        x = dxZP + (self.pattern.shape[0] // 2) + shift[0]
        y = dyZP + (self.pattern.shape[1] // 2) + shift[1]

        if x > searchedImgSize[0]:
            x -= searchedImgSize[0]
        if y > searchedImgSize[1]:
            y -= searchedImgSize[1]

        self.positionMid = (x, y)

    def updatePatternAndPosition(self, position, patternTransformed, searchedImg):
        horizSpace = self.pattern.shape[0] // 2
        verSpace = self.pattern.shape[1] // 2

        leftSide = position[0] - horizSpace
        rightSide = position[0] + horizSpace
        upperSide = position[1] - verSpace
        bottomSide = position[1] + verSpace

        filterWin = self.edgeFilter(self.pattern.shape[0])
        if leftSide < 0:
            filterWin = filterWin[:, -leftSide:]
            leftSide = 0
        if upperSide < 0:
            filterWin = filterWin[-upperSide:, :]
            upperSide = 0
        if rightSide > searchedImg.shape[0]:
            filterWin = filterWin[:, :-(rightSide - searchedImg.shape[0])]
            rightSide = searchedImg.shape[0]
        if bottomSide > searchedImg.shape[1]:
            filterWin = filterWin[:-(bottomSide - searchedImg.shape[1]), :]
            bottomSide = searchedImg.shape[1]

        partPattern = patternTransformed[upperSide:bottomSide, leftSide:rightSide]
        partSearched = searchedImg[upperSide:bottomSide, leftSide:rightSide]
        partSearched = partSearched * filterWin

        similarity = compareImgs.ssim(partPattern, partSearched)

        if similarity > 0.6:
            if partSearched.shape == self.pattern.shape:
                self.pattern = searchedImg[upperSide:bottomSide, leftSide:rightSide]
            self.positionMid = position
            self.objectIsVisible = True
        else:
            self.objectIsVisible = False

    def checkWrapedAroundPositions(self, patternTransformed, searchedImg, tolerance = 5):
        if not self.objectIsVisible:
            newPositionXY = list(self.positionMid)
            print("DEBUG 1")
            if (self.positionMid[0] - tolerance < 0) or (self.positionMid[0] + tolerance > searchedImg.shape[0]):
                newPositionXY[0] = searchedImg.shape[0] - self.positionMid[0]
                self.updatePatternAndPosition(newPositionXY, patternTransformed, searchedImg)

            if (self.positionMid[1] - tolerance < 0) or (self.positionMid[1] + tolerance > searchedImg.shape[1]):
                newPositionXY[1] = searchedImg.shape[1] - self.positionMid[1]
                self.updatePatternAndPosition(newPositionXY, patternTransformed, searchedImg)

            if self.positionMid != tuple(newPositionXY):
                self.updatePatternAndPosition(newPositionXY, patternTransformed, searchedImg)

    def initializePattern(self, patternImg, mouseXY):
        if self.pattern is None:
            self.pattern = patternImg
            self.positionGlobal = mouseXY

    def setSearchedArea(self, frame, searchRange, frameEqSearch):
        if frameEqSearch:
            self.searchedArea = frame
            return

        height, width = frame.shape

        leftSide = self.positionGlobal[0] - searchRange
        rightSide = self.positionGlobal[0] + searchRange
        upperSide = self.positionGlobal[1] - searchRange
        bottomSide = self.positionGlobal[1] + searchRange

        if np.abs(bottomSide) - np.abs(upperSide) > height:
            upperSide = 0
            bottomSide = height
            newSearchRange = int(height / 2)
            leftSide = self.positionGlobal[0] - newSearchRange
            rightSide = self.positionGlobal[0] + newSearchRange
        if np.abs(rightSide) - np.abs(leftSide) > width:
            leftSide = 0
            rightSide = width
            newSearchRange = int(width / 2)
            upperSide = self.positionGlobal[1] - newSearchRange
            bottomSide = self.positionGlobal[1] + newSearchRange

        if leftSide < 0:
            rightSide += (-leftSide)
            leftSide = 0
        elif rightSide > width:
            leftSide -= (rightSide - width)
            rightSide = width
        if upperSide < 0:
            bottomSide += (-upperSide)
            upperSide = 0
        elif bottomSide > height:
            upperSide -= (bottomSide - height)
            bottomSide = height

        self.positionGlobalShift = (leftSide, upperSide)
        self.searchedArea = frame[upperSide:bottomSide, leftSide:rightSide]

    def updatePositionGlobal(self, shift, frame, searchRange):
        globalX = self.positionGlobalShift[0] + self.positionMid[0]
        globalY = self.positionGlobalShift[1] + self.positionMid[1]
        self.positionGlobal = globalX, globalY

    def objectTracking(self, patternSection, frame, mouseXY, frameEqSearch=False):
        self.initializePattern(patternSection, mouseXY)
        self.setSearchedArea(frame, (round((self.pattern.shape[0] * 1.5) / 2)) * 2, frameEqSearch)

        pattern = self.reduceEdgeEffects(self.pattern)
        patternZP = self.imageZeroPaddingLUC(pattern, self.searchedArea.shape)

        patternZPFft = np.fft.fft2(patternZP)
        searchedFft = np.fft.fft2(self.searchedArea)

        patternZPShiftedFft = np.fft.fftshift(patternZPFft)
        searchedShiftedFft = np.fft.fftshift(searchedFft)

        patternFftMag = self.highPassFilter(patternZPShiftedFft.shape) * self.magnitude(patternZPShiftedFft)
        searchedFftMag = self.highPassFilter(searchedShiftedFft.shape) * self.magnitude(searchedShiftedFft)

        patternMagLogPolar, M = self.logPolarTransform(patternFftMag)
        searchedMagLogPolar, _ = self.logPolarTransform(searchedFftMag)

        patternMagLogPolarFft = np.fft.fft2(patternMagLogPolar)
        searchedMagLogPolarFft = np.fft.fft2(searchedMagLogPolar)

        imgsPhaseCorrMag = self.phaseCorrelationMag(patternMagLogPolarFft, searchedMagLogPolarFft)

        angles, scale = self.calculateAnglesAndScale(imgsPhaseCorrMag, searchedMagLogPolarFft.shape, M)

        img1, img2 = self.getRotatedScaledPatterns(pattern, self.searchedArea.shape, angles, scale)

        patternRotatedScaled, shift = self.bestTransformedPattern(img1, img2, searchedFft)

        patternTransformed = self.shiftImage(patternRotatedScaled, shift)

        self.predictObjectPosition(self.searchedArea.shape, shift)

        self.updatePatternAndPosition(self.positionMid, patternTransformed, self.searchedArea)
        self.checkWrapedAroundPositions(patternTransformed, self.searchedArea)

        self.updatePositionGlobal(shift, frame, self.pattern.shape[0])
