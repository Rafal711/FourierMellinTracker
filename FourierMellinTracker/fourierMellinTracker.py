import numpy as np
import cv2
import compareImgs
import filters
import imagePlot as iplt


class FourierMellinTracker:
    def __init__(self, position, edgeFilter, highPassFilter):
        self.edgeFilter = edgeFilter
        self.highPassFilter = highPassFilter
        self.positionMid = position
        self.objectIsVisible = True
        self.pattern = None

    def imageZeroPaddingLUC(self, patternImg, searchedImgSize):
        resultSize = (searchedImgSize[0] - patternImg.shape[0], searchedImgSize[1] - patternImg.shape[1])
        return cv2.copyMakeBorder(patternImg, 0, resultSize[0], 0, resultSize[1], cv2.BORDER_CONSTANT, value=0)

    def imageZeroPaddingMid(self, patternImg, searchedImgSize):
        resultSize = ((searchedImgSize[0] - patternImg.shape[0]) // 2,
                      (searchedImgSize[1] - patternImg.shape[1]) // 2)
        return np.pad(patternImg, resultSize, "constant", constant_values=0)

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

    def predictObjectPosition(self, patternSectionSize, searchedImgSize, shift):
        dxZP = (searchedImgSize[0] - patternSectionSize[0]) // 2
        dyZP = (searchedImgSize[1] - patternSectionSize[1]) // 2

        x = dxZP + (patternSectionSize[0] // 2) + shift[0]
        y = dyZP + (patternSectionSize[1] // 2) + shift[1]

        if x > searchedImgSize[0]:
            x -= searchedImgSize[0]
        if y > searchedImgSize[1]:
            y -= searchedImgSize[1]

        self.positionMid = (x, y)

    def updatePatternAndPosition(self, position, patternSectionSize, patternImg, searchedImg):
        horizSpace = patternSectionSize[0] // 2
        verSpace = patternSectionSize[1] // 2

        leftSide = position[0] - horizSpace
        rightSide = position[0] + horizSpace
        upperSide = position[1] - verSpace
        bottomSide = position[1] + verSpace

        filterWin = filters.hanning2D(patternSectionSize[0])
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

        partPattern = patternImg[upperSide:bottomSide, leftSide:rightSide]
        partSearched = searchedImg[upperSide:bottomSide, leftSide:rightSide]
        partSearched = partSearched * filterWin

        similarity = compareImgs.ssim(partPattern, partSearched)

        if similarity > 0.8:
            if partSearched.shape == searchedImg.shape:
                self.pattern = searchedImg[upperSide:bottomSide, leftSide:rightSide]
            self.positionMid = position
            self.objectIsVisible = True
        else:
            self.objectIsVisible = False

    def checkWrapedAroundPositions(self, patternSectionSize, patternImg, searchedImg, tolerance = 5):
        if not self.objectIsVisible:
            newPositionXY = list(self.positionMid)

            if (self.positionMid[0] - tolerance < 0) or (self.positionMid[0] + tolerance > searchedImg.shape[0]):
                newPositionXY[0] = searchedImg.shape[0] - self.positionMid[0]
                self.updatePatternAndPosition(newPositionXY, patternSectionSize, patternImg, searchedImg)

            if (self.positionMid[1] - tolerance < 0) or (self.positionMid[1] + tolerance > searchedImg.shape[1]):
                newPositionXY[1] = searchedImg.shape[1] - self.positionMid[1]
                self.updatePatternAndPosition(newPositionXY, patternSectionSize, patternImg, searchedImg)

            if self.positionMid != tuple(newPositionXY):
                self.updatePatternAndPosition(newPositionXY, patternSectionSize, patternImg, searchedImg)

    def objectTracking(self, patternSection, searchedSection):
        pattern = self.reduceEdgeEffects(patternSection)
        patternZP = self.imageZeroPaddingLUC(pattern, searchedSection.shape)

        patternZPFft = np.fft.fft2(patternZP)
        searchedFft = np.fft.fft2(searchedSection)

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

        img1, img2 = self.getRotatedScaledPatterns(pattern, searchedSection.shape, angles, scale)

        patternRotatedScaled, shift = self.bestTransformedPattern(img1, img2, searchedFft)

        patternTransformed = self.shiftImage(patternRotatedScaled, shift)

        self.predictObjectPosition(patternSection.shape, searchedSection.shape, shift)

        self.updatePatternAndPosition(self.positionMid, patternSection.shape, patternTransformed, searchedSection)
        self.checkWrapedAroundPositions(patternSection.shape, patternTransformed, searchedSection)

        iplt.plotImages1x2(patternTransformed, searchedSection, patternSection.shape, self.positionMid)

        #iplt.plotImage(searchedSection - patternTransformed)
