# -------------------------------------
#                IMPORTS
# -------------------------------------

import constants
import argparse
import math
import os
import shutil
import cv2
import numpy as np
from time import sleep
from tqdm import tqdm  # Loop progress bar || Resource from: https://github.com/tqdm/tqdm
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


# -----------------------------------
#              FUNCTIONS
# -----------------------------------


# ----------------------- MAKS FUNCTIONS -----------------------


def calculateMeanMasks():
    signalsMasksRed = []
    signalsMasksBlue = []

    prohibicion = constants.PROHIBICION
    peligro = constants.PELIGRO
    stop = constants.STOP
    direccionProhibida = constants.DIRECCIONPROHIBIDA
    cedaPaso = constants.CEDAPASO
    direccionObligatoria = constants.DIRECCIONOBLIGATORIA

    signs = [prohibicion, peligro, stop, direccionProhibida, cedaPaso, direccionObligatoria]

    namesListPosition = -1

    for signType in tqdm(signs):
        namesListPosition += 1
        mask = np.zeros((25, 25, 3), np.uint8)
        first = True
        for singDir in signType:
            signs = os.listdir(constants.TRAIN_PATH + '/' + singDir)
            for file in signs:
                currentResizedImage = cv2.resize(cv2.imread(constants.TRAIN_PATH + '/' + singDir + '/' + file),
                                                 (25, 25))
                if first:
                    mask = cv2.addWeighted(currentResizedImage, 1, mask, 0, 0.0)
                    first = False
                else:
                    mask = cv2.addWeighted(currentResizedImage, 0.5, mask, 0.5, 0.0)

        signalsMasksBlue.append((getColorMaskRedOrBlue(mask, 'b'), constants.SIGNALLIST[namesListPosition]))
        signalsMasksRed.append((getColorMaskRedOrBlue(mask, 'r'), constants.SIGNALLIST[namesListPosition]))

        sleep(0.01)

    return signalsMasksRed, signalsMasksBlue


# Color filtering using HSV || Resource from: https://www.geeksforgeeks.org/filter-color-with-opencv/
def getColorMaskRedOrBlue(image, color):
    imageResize = cv2.resize(image, (25, 25))
    imageHSV = cv2.cvtColor(imageResize, cv2.COLOR_BGR2HSV)

    # Red color
    if color == 'r':
        # Lower red mask
        lowerRed = np.array([0, 50, 10])
        upperRed = np.array([10, 255, 255])
        maskLower = cv2.inRange(imageHSV, lowerRed, upperRed)

        # Upper red mask
        lowerRed = np.array([160, 50, 10])
        upperRed = np.array([179, 255, 255])
        maskUpper = cv2.inRange(imageHSV, lowerRed, upperRed)

        maskRed = cv2.add(maskLower, maskUpper)

        return maskRed

    # Blue color
    elif color == 'b':
        lowerBlue = np.array([90, 70, 10], np.uint8)
        upperBlue = np.array([128, 255, 255], np.uint8)
        maskBlue = cv2.inRange(imageHSV, lowerBlue, upperBlue)

        return maskBlue


# ----------------------- MSER FUNCTIONS -----------------------


def detectSignsOnDirectory(path, mser):
    directoryDetections = []
    numberOfDetections = []
    imagesWithWindows = []
    for file in tqdm(os.listdir(path)):
        if not file.endswith('.txt'):
            detections = MSERTrafficSignDetector(cv2.imread(path + '/' + file), mser, file)
            directoryDetections.append(detections)
            numberOfDetections.append((file, len(detections)))
            image = createImageWithWindows(cv2.imread(path + '/' + file), detections)
            imagesWithWindows.append((file, image))

    sleep(0.02)
    return directoryDetections, numberOfDetections, imagesWithWindows


def MSERTrafficSignDetector(image, mser, file):
    modifiedImage = grayAndEnhanceContrast(image)

    windowsBorders = mser.detectRegions(modifiedImage)[1]

    croppedImageDetections = []
    for window in windowsBorders:

        windowCords = makeWindowBiggerOrDiscardFakeDetections(window, 1.30)
        if windowCords is not None:
            if windowCords == (890, 476, 946, 524):
                print("aqui")
            croppedImageDetections.append(
                (cv2.resize(cropImageByCoords(windowCords, image), (25, 25)), windowCords, file))

    # Clean duplicated images by pixel-similarity
    croppedImageDetections = cleanDuplicatedDetections(croppedImageDetections, False, 0.85)
    # Clean duplicated images by coordinates-similarity
    croppedImageDetections = cleanDuplicatedDetections(croppedImageDetections, True, 0.95)

    return croppedImageDetections


# Different methods combined enhances the image contrast
def grayAndEnhanceContrast(image):
    # Img turn gray
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for fine histogram equalization || Resource from:
    # https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2)
    claheImage = clahe.apply(grayImage)

    # Blur image for correct noise
    blurImage = cv2.GaussianBlur(claheImage, (3, 3), 0)

    # Enhance exposure
    imageGammaCorrection = gammaCorrection(blurImage, 2)

    result = imageGammaCorrection

    return result


def makeWindowBiggerOrDiscardFakeDetections(window, percentage):
    x1, y1, w, h = window

    x2 = x1 + w
    y2 = y1 + h

    middleDeltaW = w * (percentage - 1) * 0.5
    middleDeltaH = h * (percentage - 1) * 0.5

    squareGoodAspectRatio = True if (0.8 < w / h < 1.20) else False
    if squareGoodAspectRatio:

        x1 = x1 - middleDeltaW if x1 - middleDeltaW > 0 else 0
        y1 = y1 - middleDeltaH if y1 - middleDeltaH > 0 else 0
        x2 = x2 + middleDeltaW if x2 + middleDeltaW > 0 else 0
        y2 = y2 + middleDeltaH if y2 + middleDeltaH > 0 else 0

        return int(x1), int(y1), int(x2), int(y2)
    else:
        return None


def cleanDuplicatedDetections(imageDetections, isSimilarityByEuclideanDistanceON, tolerance):
    cleanDetections = []

    for image in imageDetections:
        image, deletions = checkIfImageIsDuplicatedOrMergeSimilarOnes(image, cleanDetections, tolerance,
                                                                      isSimilarityByEuclideanDistanceON)
        if deletions:
            for deletedImage in deletions:
                cleanDetections.pop(getElementIndexFromList(cleanDetections, deletedImage[0]))

        cleanDetections.append(image)

    return cleanDetections


def checkIfImageIsDuplicatedOrMergeSimilarOnes(image, detections, tolerance, isSimilarityByEuclideanDistanceON):
    deletions = []
    if detections:
        for detection in detections:
            if not isSimilarityByEuclideanDistanceON:

                # Use histogram comparison between two images || resource from:
                # https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
                similarity = cv2.compareHist(calculateHistAndNormalize(image[0]),
                                             calculateHistAndNormalize(detection[0]),
                                             cv2.HISTCMP_CORREL)
            else:

                # Use euclidean distance for comparison between two images using the coordinates
                imageCoords = image[1]
                detectionCoords = detection[1]

                similarity = np.sqrt(
                    EuclDSimilarity(imageCoords[0], imageCoords[1], detectionCoords[0],
                                    detectionCoords[1]) * EuclDSimilarity(imageCoords[2], imageCoords[3],
                                                                          detectionCoords[2],
                                                                          detectionCoords[3]))

            if similarity > tolerance:
                deletions.append(detection)
            elif tolerance * 0.8823 <= similarity <= tolerance:
                image = (
                    cv2.addWeighted(image[0], 0.5, detection[0], 0.5, 0.0), meanCoords(image[1], detection[1]),
                    detection[2])
                deletions.append(detection)

    return image, deletions


# ----------------------- IMAGE RECOGNITION FUNCTIONS -----------------------


def detectionsMaskCorrelation(detection, signalsMasksRed, signalsMasksBlue, tolerance):
    redHSVImageMask = getColorMaskRedOrBlue(detection[0], 'r')
    blueHSVImageMask = getColorMaskRedOrBlue(detection[0], 'b')
    scoreRed, signalIDRed = getSimilarSignalType(redHSVImageMask, signalsMasksRed)
    scoreBlue, signalIDBlue = getSimilarSignalType(blueHSVImageMask, signalsMasksBlue)

    x1, y1, x2, y2 = detection[1]
    if scoreRed > scoreBlue:
        if scoreRed > tolerance:
            return detection[2], x1, y1, x2, y2, signalIDRed, scoreRed
        else:
            return None
    else:
        if scoreBlue > tolerance:
            return detection[2], x1, y1, x2, y2, signalIDBlue, scoreBlue
        else:
            return None


def getSimilarSignalType(imageMask, signalsMasks):
    finalScore = -math.inf
    signalName = ''

    for signalMask in signalsMasks:
        signalMaskImage, signalMaskName = signalMask
        imageMaskANDSignalMask = imageMask * signalMaskImage
        score = calculateScoreBetweenMatrixs(imageMaskANDSignalMask, signalMaskImage)

        if score > finalScore:
            signalName = constants.SIGNALLIST.index(signalMaskName) + 1
            finalScore = score

    return finalScore, signalName


# ----------------------- STATISTICS FUNCTIONS -----------------------


def generateStatistics(detections, realResultsFilePath, numberDetections):
    realResults = []
    file = open(realResultsFilePath, "r")
    for line in tqdm(file):
        filename, x1, y1, x2, y2, signType = line.rstrip().split(';')
        realResults.append((filename, int(x1), int(y1), int(x2), int(y2), calculateSignType(signType)))
    print("\nResultados reales cargados desde el fichero", file)
    print("\nCalculando estadísticas...")

    detectionsPerFileByType = []
    totalDetectionsByType = {
        "prohibicion": (0, 0, 0, 0),
        "peligro": (0, 0, 0, 0),
        "stop": (0, 0, 0, 0),
        "direccionProhibida": (0, 0, 0, 0),
        "cedaPaso": (0, 0, 0, 0),
        "direccionObligatoria": (0, 0, 0, 0)
    }
    totalCorrect = 0
    totalIncorrect = 0
    totalNonDetected = 0
    expectedTotalCorrect = 0
    for fileNameAndNumber in tqdm(numberDetections):
        detectionsOnFile, realResultsOnFile = getResultsOnFile(fileNameAndNumber[0], detections, realResults)
        detectionsPerTypeOnFile, realResultsPerTypeOnFile = getResultsByTypeOnFile(detectionsOnFile, realResultsOnFile)

        totalCorrectOnFile = 0
        totalIncorrectOnFile = 0
        totalNonDetectedOnFile = 0
        expectedTotalCorrectOnFile = totalLen(realResultsPerTypeOnFile)

        signalTypeIndex = -1
        detectionsByTypeOnFileResults = []
        for detectionsByTypeOnFile, realResultsByTypeOnFile in zip(detectionsPerTypeOnFile, realResultsPerTypeOnFile):
            signalTypeIndex += 1

            totalCorrectByTypeOnFile, totalIncorrectByTypeOnFile, totalNonDetectedByTypeOnFile = getCorrectsAndWrongByTypeOnFile(
                detectionsByTypeOnFile, realResultsByTypeOnFile)
            detectionsByTypeOnFileResults.append((constants.SIGNALLIST[signalTypeIndex], totalCorrectByTypeOnFile,
                                                  totalIncorrectByTypeOnFile, totalNonDetectedByTypeOnFile,
                                                  len(realResultsByTypeOnFile)))

            totalCorrectOnFile += totalCorrectByTypeOnFile
            totalIncorrectOnFile += totalIncorrectByTypeOnFile
            totalNonDetectedOnFile += totalNonDetectedByTypeOnFile

        for detectionByTypeOnFileResults in detectionsByTypeOnFileResults:
            correct, incorrect, nonDetected, expectedCorrect = totalDetectionsByType[detectionByTypeOnFileResults[0]]
            correct += detectionByTypeOnFileResults[1]
            incorrect += detectionByTypeOnFileResults[2]
            nonDetected += detectionByTypeOnFileResults[3]
            expectedCorrect += detectionByTypeOnFileResults[4]
            totalDetectionsByType[detectionByTypeOnFileResults[0]] = (correct, incorrect, nonDetected, expectedCorrect)

        detectionsPerFileByType.append((fileNameAndNumber[0], detectionsByTypeOnFileResults, totalCorrectOnFile,
                                        totalIncorrectOnFile, totalNonDetectedOnFile, expectedTotalCorrectOnFile))

    totalDetectionsByType = list(totalDetectionsByType.items())

    for totalDetectionByType in totalDetectionsByType:
        totalCorrect += totalDetectionByType[1][0]
        totalIncorrect += totalDetectionByType[1][1]
        totalNonDetected += totalDetectionByType[1][2]
        expectedTotalCorrect += totalDetectionByType[1][3]

    return detectionsPerFileByType, totalDetectionsByType, totalCorrect, totalIncorrect, totalNonDetected, expectedTotalCorrect


def getResultsOnFile(fileName, detections, realResults):
    detectionsOnFile = []
    realResultsOnFile = []

    for detection in detections:
        if detection[0].split(".", 1)[0] == fileName.split(".", 1)[0]:
            detectionsOnFile.append(detection)

    for realResult in realResults:
        if realResult[0].split(".", 1)[0] == fileName.split(".", 1)[0]:
            realResultsOnFile.append(realResult)

    return detectionsOnFile, realResultsOnFile


def getResultsByTypeOnFile(detectionsOnFile, realResultsOnFile):
    detectionsProhibicionOnFile = []
    detectionsPeligroOnFile = []
    detectionsStopOnFile = []
    detectionsDirProhOnFile = []
    detectionsCedaPasoOnFile = []
    detectionsDirObligOnFile = []

    realResultsProhibicionOnFile = []
    realResultsPeligroOnFile = []
    realResultsStopOnFile = []
    realResultsDirProhOnFile = []
    realResultsCedaPasoOnFile = []
    realResultsDirObligOnFile = []

    detectionsByTypeOnFile = appendResultsByTypeOnFile(detectionsOnFile, detectionsProhibicionOnFile,
                                                       detectionsPeligroOnFile,
                                                       detectionsStopOnFile,
                                                       detectionsDirProhOnFile,
                                                       detectionsCedaPasoOnFile,
                                                       detectionsDirObligOnFile)

    realResultsByTypeOnFile = appendResultsByTypeOnFile(realResultsOnFile, realResultsProhibicionOnFile,
                                                        realResultsPeligroOnFile,
                                                        realResultsStopOnFile,
                                                        realResultsDirProhOnFile,
                                                        realResultsCedaPasoOnFile,
                                                        realResultsDirObligOnFile)

    return detectionsByTypeOnFile, realResultsByTypeOnFile


def appendResultsByTypeOnFile(resultsOnFile, resultsProhibicionOnFile, resultsPeligroOnFile, resultsStopOnFile,
                              resultsDirProhOnFile, resultsCedaPasoOnFile, resultsDirObligOnFile):
    for resultOnFile in resultsOnFile:
        if resultOnFile[5] == 1:
            resultsProhibicionOnFile.append(resultOnFile)
        elif resultOnFile[5] == 2:
            resultsPeligroOnFile.append(resultOnFile)
        elif resultOnFile[5] == 3:
            resultsStopOnFile.append(resultOnFile)
        elif resultOnFile[5] == 4:
            resultsDirProhOnFile.append(resultOnFile)
        elif resultOnFile[5] == 5:
            resultsCedaPasoOnFile.append(resultOnFile)
        else:
            resultsDirObligOnFile.append(resultOnFile)

    return [resultsProhibicionOnFile, resultsPeligroOnFile, resultsStopOnFile,
            resultsDirProhOnFile, resultsCedaPasoOnFile, resultsDirObligOnFile]


def getCorrectsAndWrongByTypeOnFile(detectionsByTypeOnFile, realResultsByTypeOnFile):
    checkedRealResults = set()

    correctDetections = 0
    incorrectDetections = 0
    nonDetected = 0
    duplicatedDetections = 0

    if detectionsByTypeOnFile and realResultsByTypeOnFile:
        for detectionByTypeOnFile in detectionsByTypeOnFile:
            checkResult, checkedRealResults = checkIfDetectionByTypeOnFileIsCorrectIncorrectDuplicated(
                detectionByTypeOnFile, realResultsByTypeOnFile, checkedRealResults)
            if checkResult == "correct":
                correctDetections += 1
            elif checkResult == "duplicated":
                duplicatedDetections += 1
            else:
                incorrectDetections += 1
        nonDetected += len(realResultsByTypeOnFile) - len(checkedRealResults)
    elif realResultsByTypeOnFile:
        nonDetected = len(realResultsByTypeOnFile)
    elif detectionsByTypeOnFile:
        incorrectDetections = len(detectionsByTypeOnFile)

    return correctDetections, incorrectDetections, nonDetected


def checkIfDetectionByTypeOnFileIsCorrectIncorrectDuplicated(detectionByTypeOnFile, realResultsByTypeOnFile,
                                                             checkedRealResults):
    finalEuclideanDistancesGeometricMean = -math.inf
    realResultSimilar = None
    for realResultByTypeOnFile in realResultsByTypeOnFile:
        EuclideanDistancesGeometricMean = np.sqrt(
            EuclDSimilarity(detectionByTypeOnFile[1], detectionByTypeOnFile[2], realResultByTypeOnFile[1],
                            realResultByTypeOnFile[2]) * EuclDSimilarity(detectionByTypeOnFile[3],
                                                                         detectionByTypeOnFile[4],
                                                                         realResultByTypeOnFile[3],
                                                                         realResultByTypeOnFile[4]))
        if EuclideanDistancesGeometricMean > finalEuclideanDistancesGeometricMean:
            finalEuclideanDistancesGeometricMean = EuclideanDistancesGeometricMean
            realResultSimilar = realResultByTypeOnFile

    if finalEuclideanDistancesGeometricMean > 0.85:
        checkedRealResults.add(realResultSimilar)
        return "correct", checkedRealResults
    elif finalEuclideanDistancesGeometricMean > 0.85 and realResultSimilar in checkedRealResults:
        return "duplicated", checkedRealResults
    else:
        return "incorrect", checkedRealResults


# ----------------------- UTILS FUNCTIONS -----------------------


# Percentual function (based on sigmoid function) uses euclidean distance between 2 points (The closer values are,
# this returns values close to 1. At determined distance called "closeness limit", points are considered far,
# so the function returns values close to 0)... Many hours trying values on Geogebra
def EuclDSimilarity(xA, yA, xB, yB):
    euclDist = np.linalg.norm(np.array((xA, yA)) - np.array((xB, yB)))
    return 1 / (1 + np.power(np.e,
                             (((0.154 * np.power(euclDist, 1.2)) - 31.8) / (0.2 * euclDist)))) if euclDist > 0 else 1


def meanCoords(coordsA, coordsB):
    x1A, y1A, x2A, y2A = coordsA
    x1B, y1B, x2B, y2B = coordsB
    return (x1A + x1B) // 2, (y1A + y1B) // 2, (x2A + x2B) // 2, (y2A + y2B) // 2


def getElementIndexFromList(l, element):
    # Consider "l" contains "element"
    index = 0
    for x in l:
        if np.array_equal(x[0], element):
            return index
        index += 1


def precision(truePositives, falsePositives):
    if truePositives > 0 or falsePositives > 0:
        return round(truePositives / (truePositives + falsePositives), 2)
    else:
        return "NaN"


def recall(truePositives, falseNegatives):
    if truePositives > 0 or falseNegatives > 0:
        return round(truePositives / (truePositives + falseNegatives), 2)
    else:
        return "NaN"


def score(truePositives, falsePositives, falseNegatives):
    if truePositives > 0 or falsePositives > 0 or falseNegatives > 0:
        return round((2 * truePositives) / ((2 * truePositives) + falsePositives + falseNegatives), 2)
    else:
        return "NaN"


def createDetectionsStrings(detections):
    detectionsStrings = []
    for detection in detections:
        filename, x1, y1, x2, y2, signType, score = detection
        detectionsStrings.append(
            filename + ";" + str(x1) + ";" + str(y1) + ";" + str(x2) + ";" + str(y2) + ";" + str(signType) + ";" + str(
                score))
    return detectionsStrings


def totalLen(l):
    total = 0
    for x in l:
        total += len(x)
    return total


def calculateSignType(signType):
    prohibicion = constants.PROHIBICION
    peligro = constants.PELIGRO
    stop = constants.STOP
    direccionProhibida = constants.DIRECCIONPROHIBIDA
    cedaPaso = constants.CEDAPASO
    direccionObligatoria = constants.DIRECCIONOBLIGATORIA

    if int(signType) < 10:
        signType = '0' + signType

    if signType in prohibicion:
        return 1
    elif signType in peligro:
        return 2
    elif signType in stop:
        return 3
    elif signType in direccionProhibida:
        return 4
    elif signType in cedaPaso:
        return 5
    elif signType in direccionObligatoria:
        return 6


# based on recall and precision calculus || Resource from:
# https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
def calculateScoreBetweenMatrixs(matrix1, matrix2):
    truePositives = 0
    falsePositives = 0
    falseNegatives = 0
    trueNegatives = 0

    if matrix1.shape == matrix2.shape:
        matrix2 = matrix2 // 255
        for rowMatrix1, rowMatrix2 in zip(matrix1, matrix2):
            for elementMatrix1, elementMatrix2 in zip(rowMatrix1, rowMatrix2):
                if elementMatrix1 == 1 and elementMatrix2 == 1:
                    truePositives += 1
                elif elementMatrix1 == 1 and elementMatrix2 == 0:
                    falsePositives += 1
                elif elementMatrix1 == 0 and elementMatrix2 == 1:
                    falseNegatives += 1
                else:
                    trueNegatives += 1
        matrixShape = matrix1.shape[0] * matrix1.shape[1]
        if matrixShape + matrixShape * 0.01 >= trueNegatives >= matrixShape - matrixShape * 0.01:
            return 0
        else:
            return round((2 * truePositives) / ((2 * truePositives) + falsePositives + falseNegatives), 2)


def cropImageByCoords(coords, image):
    x1, y1, x2, y2 = coords
    return image[y1:y2, x1:x2]


def calculateHistAndNormalize(image):
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    histSize = [50, 60]
    HueRanges = [0, 180]
    SaturationRanges = [0, 256]
    ranges = HueRanges + SaturationRanges
    channels = [0, 1]

    imageHist = cv2.calcHist([imageHSV], channels, None, histSize, ranges, accumulate=False)

    return cv2.normalize(imageHist, imageHist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)


def createImageWithWindows(image, windowsBorders):
    for detectedImage in windowsBorders:
        x1, y1, x2, y2 = detectedImage[1]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return image


# Gamma correction for enhance exposure || Resource from:
# https://lindevs.com/apply-gamma-correction-to-an-image-using-opencv/
def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

# ----------------------- OTHER ALTERNATIVES FOR RECOGNITION -----------------------

# Feature vector using Haar-like-features
def featureVectorHLF(image):
    integralImage = integral_image(image)
    return haar_like_feature(integralImage, 0, 0, integralImage.shape[0], integralImage.shape[1],featureType='type-4',featureCoord=None)

# Dimensionality reduction algorithms whit PCA
def pca(n_components):
    return PCA(n_components, svd_solver='full')

# KNN classifier
def knnClassifier ():
    return KNeighborsClassifier(n_neighbors=4)


# ----------------------- TEST -----------------------


def test(trainPath, testPath, MSERValues):
    constants.TRAIN_PATH = trainPath
    constants.TEST_PATH = testPath

    print("\nVa a comenzar el test de detección de señales de tráfico...")
    print("\nGenerando mascaras a partir de imágenes de entrenamiento... \n(" + constants.TRAIN_PATH + ")")

    try:
        meanMasks = calculateMeanMasks()
    except Exception as e:
        print("Ha ocurrido un problema generando las máscaras :(")
        print("\n"
              "------------------------------------------------------------\n"
              "                        TEST FALLIDO                        \n"
              "------------------------------------------------------------\n")
        print(e)
    else:
        print("Máscaras generadas con éxito")
        print("\nIniciando detector MSER...")

        try:
            delta, minA, maxA, maxVar = MSERValues

            delta = delta
            minArea = minA
            maxArea = maxA
            maxVariation = maxVar

            mser = cv2.MSER_create(delta=delta, min_area=minArea, max_area=maxArea, max_variation=maxVariation)
        except Exception as e:
            print("Ha ocurrido un error generando el detector :(")
            print("\n"
                  "------------------------------------------------------------\n"
                  "                        TEST FALLIDO                        \n"
                  "------------------------------------------------------------\n")
            print(e)
        else:
            print("Se ha creado con éxito el detector MSER con parámetros:\n")
            print("   DELTA:", delta)
            print("   MIN AREA:", minArea)
            print("   MAX AREA:", maxArea)
            print("   MAX VARIATION:", maxVariation)

            print(
                "\nVa a comenzar la detección de señales de tráfico en las imágenes de test... \n(" + constants.TEST_PATH + ")\n")
            print("Analizando y extrayendo regiones de interés...")

            try:
                detections, numDetections, imagesWithDetections = detectSignsOnDirectory(constants.TEST_PATH, mser)
            except Exception as e:
                print("Ha ocurrido un error en el proceso de detección de señales :(")
                print("\n"
                      "------------------------------------------------------------\n"
                      "                        TEST FALLIDO                        \n"
                      "------------------------------------------------------------\n")
                print(e)
            else:
                resultImagesPath = "resultado_imgs"

                print("\nEl proceso ha concluido con éxito, las imágenes de test con sus respectivas detecciones (con "
                      "eliminación de repeticiones) serán "
                      "guardadas en", resultImagesPath)

                print("\nGenerando resultados...")

                try:
                    if os.path.isdir(resultImagesPath):
                        shutil.rmtree(resultImagesPath)

                    os.mkdir(resultImagesPath)

                    for file, image in tqdm(imagesWithDetections):
                        cv2.imwrite(resultImagesPath + "/" + file, image)
                        sleep(0.02)
                except Exception as e:
                    print("Ha ocurrido un error guardando los archivos :(")
                    print("\n"
                          "------------------------------------------------------------\n"
                          "                        TEST FALLIDO                        \n"
                          "------------------------------------------------------------\n")
                    print(e)
                else:
                    print(
                        "\nA continuación se listarán las detecciones obtenidas, con eliminación de repeticiones, "
                        "por cada "
                        "archivo en",
                        constants.TEST_PATH + "\n")

                    total = 0
                    for file, number in numDetections:
                        print(file, ".......", number, "   detecciones" if number < 10 else "  detecciones")
                        total += number
                    print("Total ...........", total, "detecciones")

                    print("\nVa a comenzar el proceso de filtrado por correlación de máscaras...\n")
                    print("Realizando el filtrado...")
                    try:
                        detectionResults = []
                        for detectionsPerFile in tqdm(detections):
                            for detection in detectionsPerFile:
                                detectionMaskCorrelationResult = detectionsMaskCorrelation(detection, meanMasks[0],
                                                                                           meanMasks[1], 0.55)
                                if detectionMaskCorrelationResult is not None:
                                    detectionResults.append(detectionMaskCorrelationResult)

                        detections = detectionResults
                    except Exception as e:
                        print("Ha ocurrido un error en el proceso de correlación de máscaras :(")
                        print("\n"
                              "------------------------------------------------------------\n"
                              "                        TEST FALLIDO                        \n"
                              "------------------------------------------------------------\n")
                        print(e)
                    else:
                        finalDetectionsPath = "resultado.txt"

                        print("\nEl proceso ha concluido con éxito y las detecciones filtradas serán guardadas en el "
                              "archivo", finalDetectionsPath + ". La codificación de los resultados es la siguiente:")
                        print("nombre_archivo.jpg;x1_coord;y1_coord;x2_coord;y2_coord;tipo_señal;score\n")
                        print("  x1_coord  ===>  coordenada x de la esquina superior izquierda de la detección\n"
                              "  y1_coord  ===>  coordenada y de la esquina superior izquierda de la detección\n"
                              "  x2_coord  ===>  coordenada x de la esquina inferior derecha de la detección\n"
                              "  y2_coord  ===>  coordenada y de la esquina inferior derecha de la detección\n"
                              "tipo_señal  ===>  1 -> Obligación      2 -> Peligro        3 -> Stop\n"
                              "                  4 -> dir Prohibida   5 -> ceda el paso   6 -> dir obligatoria\n"
                              "     score  ===>  Puntuación de acierto en la detección\n")

                        print("Generando archivo de resultados...")

                        try:
                            detectionsStrings = createDetectionsStrings(detections)
                            file = open(finalDetectionsPath, "w")
                            for detection in tqdm(detectionsStrings):
                                file.write(detection + "\n")
                            file.close()
                        except Exception as e:
                            print("Ha ocurrido un error en el proceso de creación del archivo de resultados :(")
                            print("\n"
                                  "------------------------------------------------------------\n"
                                  "                        TEST FALLIDO                        \n"
                                  "------------------------------------------------------------\n")
                            print(e)
                        else:
                            realResultsFilePath = "test_alumnos_jpg/gt.txt"
                            print("\nEl archivo de resultados ha sido generado correctamente")
                            print(
                                "\nFinalmente van a mostrarse las estadísticas de funcionamiento del programa a "
                                "partir del fichero de resultados reales",
                                realResultsFilePath)
                            try:
                                print("\nGenerando estadísticas...")

                                statisticsResults = generateStatistics(detections, realResultsFilePath, numDetections)
                            except Exception as e:
                                print("Ha ocurrido un error generando las estadísticas :(")
                                print("\n"
                                      "------------------------------------------------------------\n"
                                      "                        TEST FALLIDO                        \n"
                                      "------------------------------------------------------------\n")
                                print(e)
                            else:
                                print("\nLas estadísticas de funcionamiento son las siguientes:")

                                detectionsPerFileByType, totalDetectionsByType, totalCorrect, totalIncorrect, totalNonDetected, expectedTotalCorrect = statisticsResults

                                print("\n"
                                      "---------------------------------\n"
                                      "     DETECCIONES POR ARCHIVO     \n"
                                      "---------------------------------\n")
                                for detectionPerFile in detectionsPerFileByType:
                                    fileName, detectionsByTypeOnFile, totalCorrectOnFile, totalIncorrectOnFile, totalNonDetectedOnFile, expectedTotalCorrectOnFile = detectionPerFile
                                    print(fileName,
                                          "..................................................................")
                                    print("                TOTAL DETECTADAS CORRECTAS:", totalCorrectOnFile)
                                    print("              TOTAL DETECTADAS INCORRECTAS:", totalIncorrectOnFile)
                                    print("                       TOTAL NO DETECTADAS:", totalNonDetectedOnFile)
                                    print("      VALOR DETECTADAS CORRECTAS ESPERADAS:", expectedTotalCorrectOnFile)
                                    print("                                 PRECISIÓN:",
                                          precision(totalCorrectOnFile, totalIncorrectOnFile))
                                    print("                           TASA DE ACIERTO:",
                                          recall(totalCorrectOnFile, totalNonDetectedOnFile))
                                    print("                                PUNTUACIÓN:",
                                          score(totalCorrectOnFile, totalIncorrectOnFile, totalNonDetectedOnFile))
                                    for detectionByTypeOnFile in detectionsByTypeOnFile:
                                        signType, totalCorrectByTypeOnFile, totalIncorrectByTypeOnFile, totalNonDetectedByTypeOnFile, expectedTotalCorrectByTypeOnFile = detectionByTypeOnFile
                                        print(
                                            "\n            " + signType + ":..................................................................")
                                        print("                                     TOTAL DETECTADAS CORRECTAS:",
                                              totalCorrectByTypeOnFile)
                                        print("                                   TOTAL DETECTADAS INCORRECTAS:",
                                              totalIncorrectByTypeOnFile)
                                        print("                                            TOTAL NO DETECTADAS:",
                                              totalNonDetectedByTypeOnFile)
                                        print("                           VALOR DETECTADAS CORRECTAS ESPERADAS:",
                                              expectedTotalCorrectByTypeOnFile)
                                        print("                                                      PRECISIÓN:",
                                              precision(totalCorrectByTypeOnFile, totalIncorrectByTypeOnFile))
                                        print("                                                TASA DE ACIERTO:",
                                              recall(totalCorrectByTypeOnFile, totalNonDetectedByTypeOnFile))
                                        print("                                                     PUNTUACIÓN:",
                                              score(totalCorrectByTypeOnFile, totalIncorrectByTypeOnFile,
                                                    totalNonDetectedByTypeOnFile))

                                print("\n"
                                      "---------------------------------------\n"
                                      "     DETECCIONES POR TIPO DE SEÑAL     \n"
                                      "---------------------------------------\n")
                                for detectionByType in totalDetectionsByType:
                                    signType = detectionByType[0]
                                    totalCorrectByType, totalIncorrectByType, totalNonDetectedByType, expectedTotalCorrectByType = \
                                        detectionByType[1]
                                    print(signType,
                                          "..................................................................")
                                    print("                TOTAL DETECTADAS CORRECTAS:", totalCorrectByType)
                                    print("              TOTAL DETECTADAS INCORRECTAS:", totalIncorrectByType)
                                    print("                       TOTAL NO DETECTADAS:", totalNonDetectedByType)
                                    print("      VALOR DETECTADAS CORRECTAS ESPERADAS:", expectedTotalCorrectByType)
                                    print("                                 PRECISIÓN:",
                                          precision(totalCorrectByType, totalIncorrectByType))
                                    print("                           TASA DE ACIERTO:",
                                          recall(totalCorrectByType, totalNonDetectedByType))
                                    print("                                PUNTUACIÓN:",
                                          score(totalCorrectByType, totalIncorrectByType, totalNonDetectedByType))

                                print("\n"
                                      "-----------------------------\n"
                                      "     DETECCIONES TOTALES     \n"
                                      "-----------------------------\n")
                                print("          TOTAL DETECTADAS CORRECTAS:", totalCorrect)
                                print("        TOTAL DETECTADAS INCORRECTAS:", totalIncorrect)
                                print("                 TOTAL NO DETECTADAS:", totalNonDetected)
                                print("VALOR DETECTADAS CORRECTAS ESPERADAS:", expectedTotalCorrect)
                                print("                           PRECISIÓN:",
                                      precision(totalCorrect, totalIncorrect))
                                print("                     TASA DE ACIERTO:",
                                      recall(totalCorrect, totalNonDetected))
                                print("                          PUNTUACIÓN:",
                                      score(totalCorrect, totalIncorrect, totalNonDetected))

                                print("\n"
                                      "------------------------------------------------------------\n"
                                      "                  TEST TERMINADO CON ÉXITO                  \n"
                                      "------------------------------------------------------------\n")