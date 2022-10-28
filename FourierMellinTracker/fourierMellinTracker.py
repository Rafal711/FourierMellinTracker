import matplotlib.pyplot as plt
import numpy as np
import cv2
#from mpl_toolkits import mplot3d


class FourierMellinTracker:
    def __init__(self, edgeFilter, highPassFilter):
        self.edgeFilter = edgeFilter
        self.highPassFilter = highPassFilter

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

    def plotShiftedAndSearchedImg(self, img, shift, searchedImg):
        shiftedImg = np.roll(img, shift[0], axis=1)  # horizontally
        shiftedImg = np.roll(shiftedImg, shift[1], axis=0)  # vertically

        plt.figure(figsize=(12, 7))

        plt.subplot(1, 2, 1)
        plt.imshow(shiftedImg)
        plt.gray()
        plt.title("pattern scaled and rotated")

        plt.subplot(1, 2, 2)
        plt.imshow(searchedImg)
        plt.gray()
        plt.title("searched image")
        plt.show()

    def plotImage(self, img):
        fig, ax = plt.subplots()
        ax.imshow(img, 'gray')
        ax.axis('off')
        plt.show()

    def plot3dImage(self, img):
        ax = plt.axes(projection='3d')
        xx, yy = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
        X, Y = xx.ravel(), yy.ravel()
        Z = np.zeros_like(img).ravel()
        dx = .25 * np.ones(img.size)
        dy = .25 * np.ones(img.size)
        dz = img.ravel()
        ax.bar3d(X, Y, Z, dx, dy, dz, color = 'w')
        plt.show()

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

        #self.plot3dImage(imgsPhaseCorrMag)
        self.plotShiftedAndSearchedImg(patternRotatedScaled, shift, searchedSection)
