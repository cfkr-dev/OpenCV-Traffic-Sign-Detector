# -------------------------------------------------------------------------------------------------
#   Universidad Rey Juan Carlos - Grado en Ingeniería Informática - Visión Artificial
#
#       Práctica 2 - Reconocimiento de Objetos
#
#       Desarrollado por:
#             - Alberto Pérez Pérez (GII + GIS)
#             - Daniel Tolosa Oropesa (GII)
# -------------------------------------------------------------------------------------------------


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


# -----------------------------------
#              FUNCTIONS
# -----------------------------------

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

        windowCords = makeWindowBiggerOrDiscardFakeDetections(window, 1.15)
        if windowCords is not None:
            if windowCords == (890, 476, 946, 524):
                print("aqui")
            croppedImageDetections.append(
                (cv2.resize(cropImageByCoords(windowCords, image), (25, 25)), windowCords, file, 0))

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


# Intersection over union algorithm || Resource form:
# https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def intersectionOverUnion(imageACoords, imageBCoords):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(imageACoords[0], imageBCoords[0])
    yA = max(imageACoords[1], imageBCoords[1])
    xB = min(imageACoords[2], imageBCoords[2])
    yB = min(imageACoords[3], imageBCoords[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    imageACoordsArea = (imageACoords[2] - imageACoords[0] + 1) * (imageACoords[3] - imageACoords[1] + 1)
    imageBCoordsArea = (imageBCoords[2] - imageBCoords[0] + 1) * (imageBCoords[3] - imageBCoords[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(imageACoordsArea + imageBCoordsArea - interArea)
    # return the intersection over union value
    return iou


# ----------------------- TRAIN DATA LOADING FUNCTIONS -----------------------

def loadTrainRealResults(path):
    realResults = []
    file = open(path, "r")
    for line in tqdm(file):
        filename, x1, y1, x2, y2, signType = line.rstrip().split(';')
        realResults.append(
            (filename.split('.')[0] + '.jpg', int(x1), int(y1), int(x2), int(y2), calculateSignType(signType)))
    file.close()
    return realResults


def loadTrainImages(path):
    trainImages = []
    filesOnDir = os.listdir(path)
    for file in filesOnDir:
        if file.endswith('.jpg'):
            trainImages.append((cv2.imread(path + '/' + file), file))
    return trainImages


def OrderTrainResultsCroppedImagesByImageFile(trainImages, trainResults):
    trainCroppedImagesOrderByImageFile = dict((imageFileName, []) for imageFileName in trainImages)
    for trainResult in trainResults:
        trainResultCoords = trainResult[1], trainResult[2], trainResult[3], trainResult[4]
        resizedCropImage = cv2.resize(cropImageByCoords(trainResultCoords, cv2.imread(constants.TRAIN_PATH + '/' + trainResult[0])), (32, 32))
        trainCroppedImagesOrderByImageFile[trainResult[0]].append((resizedCropImage, trainResultCoords, trainResult[0], trainResult[5]))
    return trainCroppedImagesOrderByImageFile


def calculateNegativeTrainResults(trainImages, positiveTrainResults, mser):
    # calculate all windows of all images with MSER detector from project 1
    allImagesMSERDetections = []
    for image in trainImages:
        MSERDetections = MSERTrafficSignDetector(image[0], mser, image[1])
        allImagesMSERDetections.extend(MSERDetections)
    allImagesMSERDetectionsOrderByImageFile = OrderTrainResultsCroppedImagesByImageFile(trainImages, allImagesMSERDetections)

    # use intersection over union to calculate negative results (IoU <= 0.2)
    negativeTrainResults = dict((imageFileName, []) for imageFileName in trainImages)
    for image in trainImages:
        imageFileName = image[1]
        for detectedImage in allImagesMSERDetectionsOrderByImageFile[imageFileName]:
            lastScore = -math.inf
            for positiveTrainResult in positiveTrainResults[imageFileName]:
                intersectionOverUnionScore = intersectionOverUnion(detectedImage[1], positiveTrainResult[1])
                if intersectionOverUnionScore >= lastScore:
                    lastScore = intersectionOverUnionScore
            if lastScore <= 0.2:
                negativeTrainResults[imageFileName].append(detectedImage)
    return negativeTrainResults


def calculateHOGDescriptors(images, trainImages):
    imagesHOGDescriptors = dict((imageFileName, []) for imageFileName in trainImages)
    for image in trainImages:
        for detection in images[image[1]]:
            imagesHOGDescriptors[image[1]].append()
    pass


def loadTrainData(mser):
    trainImages = loadTrainImages(constants.TRAIN_PATH)
    trainResults = loadTrainRealResults(constants.TRAIN_PATH_REAL_RESULTS)
    positiveTrainResults = OrderTrainResultsCroppedImagesByImageFile(trainImages, trainResults)
    negativeTrainResults = calculateNegativeTrainResults(trainImages, positiveTrainResults, mser)
    positiveTrainResultsHOGDescriptors = calculateHOGDescriptors(positiveTrainResults, trainImages)
    negativeTrainResultsHOGDescriptors = calculateHOGDescriptors(negativeTrainResults, trainImages)
    return positiveTrainResults, negativeTrainResults, positiveTrainResultsHOGDescriptors, negativeTrainResultsHOGDescriptors

# --------------------------------------
#              MAIN PROGRAM
# --------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Entrena sober train y ejecuta el clasificador sobre imgs de test')
    parser.add_argument(
        '--train_path', type=str, default="./train", help='Path al directorio de imgs de train')
    parser.add_argument(
        '--test_path', type=str, default="./test", help='Path al directorio de imgs de test')
    parser.add_argument(
        '--classifier', type=str, default="BAYES", help='String con el nombre del clasificador')

    args = parser.parse_args()

    # Cargar los datos de entrenamiento 
    # args.train_path

    # Tratamiento de los datos

    # Crear el clasificador 
    if args.classifier == "BAYES":
        # detector = ...
        None
    else:
        raise ValueError('Tipo de clasificador incorrecto')

    # Entrenar el clasificador si es necesario ...
    # detector ...

    # Cargar y procesar imgs de test 
    # args.train_path ...

    # Guardar los resultados en ficheros de texto (en el directorio donde se 
    # ejecuta el main.py) tal y como se pide en el enunciado.
