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


import math
import os
import pickle
import random
from time import sleep

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm  # Loop progress bar || Resource from: https://github.com/tqdm/tqdm

import constants


# -----------------------------------
#              FUNCTIONS
# -----------------------------------

# ----------------------- MSER FUNCTIONS -----------------------

def initializeMSER(mserParams):
    delta, minArea, maxArea, maxVariation = mserParams
    mser = cv2.MSER_create(delta=delta, min_area=minArea, max_area=maxArea, max_variation=maxVariation)
    return mser


def MSERTrafficSignDetector(image, mser, file):
    modifiedImage = grayAndEnhanceContrast(image)

    windowsBorders = mser.detectRegions(modifiedImage)[1]

    croppedImageDetections = []
    for window in windowsBorders:
        windowCords = makeWindowBiggerOrDiscardFakeDetections(window, 1.15)
        if windowCords is not None:
            croppedImageDetections.append(
                (cv2.resize(cropImageByCoords(windowCords, image), (32, 32)), windowCords, file, 0))

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
                    detection[2], detection[3])
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


def loadImages(path):
    images = {}
    filesOnDir = os.listdir(path)
    for file in tqdm(filesOnDir):
        if file.endswith('.jpg'):
            images[file] = cv2.imread(path + '/' + file)
    sleep(0.02)
    return images


def orderCroppedImagesByImageFile(trainImages, trainResults):
    trainCroppedImagesOrderByImageFile = dict((imageFileName, []) for imageFileName in trainImages.keys())
    for trainResult in tqdm(trainResults):
        trainResultCoords = trainResult[1], trainResult[2], trainResult[3], trainResult[4]
        resizedCropImage = cv2.resize(
            cropImageByCoords(trainResultCoords, cv2.cvtColor(trainImages[trainResult[0]], cv2.COLOR_BGR2GRAY)), (32, 32))
        trainCroppedImagesOrderByImageFile[trainResult[0]].append(
            (resizedCropImage, trainResultCoords, trainResult[0], trainResult[5]))
    sleep(0.02)
    return trainCroppedImagesOrderByImageFile


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


def getElementIndexFromList(l, element):
    # Consider "l" contains "element"
    index = 0
    for x in l:
        if np.array_equal(x[0], element):
            return index
        index += 1


def randomBinaryArray(zerosNumber, noZerosNumber, signType):
    arr = np.zeros(zerosNumber)
    arr[:noZerosNumber] = signType
    np.random.shuffle(arr)
    return arr


def randomMultiArray(data):
    arr = []
    for signType in range(0, 7):
        arr.extend([signType] * len(data[signType]))
    np.random.shuffle(arr)
    return arr


def extractDescriptorsAndRealSignTypes(detectionsDescriptors):
    descriptors = []
    realSignTypes = []
    for detection in detectionsDescriptors:
        descriptors.append(detection[0])
        realSignTypes.append(detection[3])
    return descriptors, realSignTypes


def flatData(data):
    dataFlat = []
    for x in tqdm(data):
        dataFlat.extend(x)
    sleep(0.02)
    return dataFlat


def pickDataBySignType(negativeDataDescriptors, trainDataDescriptorByType, randomTagsBySignType):
    data = []
    for tag in randomTagsBySignType:
        if tag == 0:
            sample = negativeDataDescriptors.pop()
        else:
            sample = trainDataDescriptorByType.pop()
        data.append(sample[0])
    return data


def multiPickDataBySignType(dataDescriptors, randomTagsAllSignTypes):
    data = []
    dataDescriptorsCopy = [dataDescriptors[signType][:] for signType in range(0, 7)]
    for tag in randomTagsAllSignTypes:
        data.append(dataDescriptorsCopy[tag].pop()[0])
    return data

def noSignDetected(bestInstancePredictions, tolerance):
    for instance in bestInstancePredictions:
        if instance[1] != 0:
            if instance[0] > tolerance:
                return False
    return True


# ----------------------- TRAIN DATA LOADING FUNCTIONS -----------------------

def loadTrainRealResults(path):
    realResults = []
    file = open(path, "r")
    for line in tqdm(file):
        filename, x1, y1, x2, y2, signType = line.rstrip().split(';')
        codifiedSignType = calculateSignType(signType)
        if codifiedSignType is not None:
            realResults.append((filename.split('.')[0] + '.jpg', int(x1), int(y1), int(x2), int(y2), codifiedSignType))
    sleep(0.02)
    file.close()
    return realResults


def computeNegativeTrainResults(trainImages, positiveTrainResults, allImagesMSERDetections):
    negativeTrainResults = dict((imageFileName, []) for imageFileName in trainImages.keys())
    for fileName in tqdm(trainImages.keys()):
        for detectedImage in allImagesMSERDetections[fileName]:
            lastScore = -math.inf
            for positiveTrainResult in positiveTrainResults[fileName]:
                intersectionOverUnionScore = intersectionOverUnion(detectedImage[1], positiveTrainResult[1])
                if intersectionOverUnionScore >= lastScore:
                    lastScore = intersectionOverUnionScore
            if lastScore <= 0.5:
                negativeTrainResults[fileName].append(detectedImage)
    sleep(0.02)
    return negativeTrainResults


def calculateNegativeTrainResults(trainImages, positiveTrainResults, mser):
    if not os.path.exists('MSERTrain.val'):
        allImagesMSERDetections = dict((imageFileName, []) for imageFileName in trainImages.keys())
        print("\nExtrayendo detecciones mediante MSER...")
        for fileName, image in tqdm(trainImages.items()):
            MSERDetections = MSERTrafficSignDetector(image, mser, fileName)
            MSERDetectionsGray = []
            for detection in MSERDetections:
                MSERDetectionsGray.append((cv2.cvtColor(detection[0], cv2.COLOR_BGR2GRAY), detection[1], detection[2], detection[3]))
            allImagesMSERDetections[fileName] = MSERDetectionsGray
        sleep(0.02)
        with open("MSERTrain.val", "wb") as outfile:
            pickle.dump(allImagesMSERDetections, outfile)
    else:
        with open("MSERTrain.val", "rb") as infile:
            allImagesMSERDetections = pickle.load(infile)
    print("\nCalculando conjunto de detecciones correspondientes a no señal...")
    negativeTrainResults = computeNegativeTrainResults(trainImages, positiveTrainResults, allImagesMSERDetections)
    return negativeTrainResults


def extractDataOrderBySignType(trainResultsOrderByImageFile, signType):
    data = []
    for imageFileName in trainResultsOrderByImageFile.keys():
        for detection in trainResultsOrderByImageFile[imageFileName]:
            if detection[3] == signType:
                data.append(detection)
    return data


def formatTrainDataBySignType(positiveTrainResultsOrderByImageFile, negativeTrainResultsOrderByImageFile):
    trainDataBySignType = dict((signType, []) for signType in range(0, 7))
    for signType in tqdm(range(0, 7)):
        if signType == 0:
            trainDataBySignType[signType] = extractDataOrderBySignType(negativeTrainResultsOrderByImageFile, signType)
        else:
            trainDataBySignType[signType] = extractDataOrderBySignType(positiveTrainResultsOrderByImageFile, signType)
        random.shuffle(trainDataBySignType[signType])
    sleep(0.02)
    return trainDataBySignType


def calculateTrainDataOrderBySignType(trainImages, trainResults, mser):
    print("\nExtrayendo resultados positivos desde los datos de entrenamiento...")
    positiveTrainResultsOrderByImageFile = orderCroppedImagesByImageFile(trainImages, trainResults)
    print("\nExtrayendo resultados negativos (mediante detecciones con MSER) desde los datos de entrenamiento...")
    negativeTrainResultsOrderByImageFile = calculateNegativeTrainResults(trainImages,
                                                                         positiveTrainResultsOrderByImageFile, mser)
    print("\nAplicando formato...")
    trainDataOrderBySignType = formatTrainDataBySignType(positiveTrainResultsOrderByImageFile,
                                                         negativeTrainResultsOrderByImageFile)
    return trainDataOrderBySignType


def loadTrainData(mser):
    try:
        print("\nCargando imágenes de entrenamiento...\n(" + constants.TRAIN_PATH + ")\n")
        trainImages = loadImages(constants.TRAIN_PATH)
    except Exception as e:
        print("Ha ocurrido un error cargando las imágenes de entrenamiento :(")
        print("\n"
              "------------------------------------------------------------\n"
              "                        TEST FALLIDO                        \n"
              "------------------------------------------------------------\n")
        print(e)
    else:
        try:
            print("\nCargando resultados reales de entrenamiento...\n(" + constants.TRAIN_PATH_REAL_RESULTS + ")\n")
            trainResults = loadTrainRealResults(constants.TRAIN_PATH_REAL_RESULTS)
        except Exception as e:
            print("Ha ocurrido un error cargando los resultados reales de entrenamiento :(")
            print("\n"
                  "------------------------------------------------------------\n"
                  "                        TEST FALLIDO                        \n"
                  "------------------------------------------------------------\n")
            print(e)
        else:
            try:
                print(
                    "\nExtrayendo resultados positivos y negativos (mediante detecciones con MSER) desde los datos de entrenamiento...")
                trainDataOrderBySignType = calculateTrainDataOrderBySignType(trainImages, trainResults, mser)
            except Exception as e:
                print(
                    "Ha ocurrido un error extrayendo los resultados positivos y negativos desde los datos de entrenamiento :(")
                print("\n"
                      "------------------------------------------------------------\n"
                      "                        TEST FALLIDO                        \n"
                      "------------------------------------------------------------\n")
                print(e)
            else:
                return trainDataOrderBySignType, trainImages


# ----------------------- EVALUATION DATA LOADING FUNCTIONS -----------------------

def extractEvaluationTestResults(trainResults, percentage):
    trainDetectionsOrderBySignType = dict((signType, []) for signType in range(0, 7))
    testDetectionsOrderBySignType = dict((signType, []) for signType in range(0, 7))
    for signType in tqdm(range(0, 7)):
        trainDetectionsOrderBySignType[signType], testDetectionsOrderBySignType[signType] = train_test_split(
            trainResults[signType], shuffle=False, test_size=percentage)
    sleep(0.02)
    return trainDetectionsOrderBySignType, testDetectionsOrderBySignType


# ----------------------- HOG FUNCTIONS -----------------------

def initializeHOG(hogParams, featureDescriptor):
    if featureDescriptor == 'HOG':
        winSize, blockSize, blockStride, cellSize, nbins, signedGradient = hogParams
        hog = cv2.HOGDescriptor(_winSize=winSize, _blockSize=blockSize, _blockStride=blockStride, _cellSize=cellSize,
                                _nbins=nbins, _signedGradient=signedGradient)
        return hog
    else:
        return None


# ----------------------- FEATURES DESCRIPTORS FUNCTIONS -----------------------

def initializeFeatureDescriptor(featureDescriptor):
    featureDescriptorInitializations = {
        'HOG': initializeHOG(constants.HOG_FEATURE_DESCRIPTORS_PARAMS[0], featureDescriptor),
        'GRAY': None
    }
    return featureDescriptorInitializations[featureDescriptor], featureDescriptor


def calculateDescriptors(trainImages, featureDescriptor):
    imagesFeatureDescriptors = dict((signType, []) for signType in range(0, 7))
    for signType in tqdm(trainImages.keys()):
        for detection in trainImages[signType]:
            imagesFeatureDescriptors[signType].append(
                (computeDescriptors(detection[0], featureDescriptor), detection[1], detection[2], detection[3]))
    sleep(0.02)
    return imagesFeatureDescriptors


def computeDescriptors(image, featureDescriptor):
    if featureDescriptor[1] == 'HOG':
        return featureDescriptor[0].compute(image)
    elif featureDescriptor[1] == 'GRAY':
        return image.ravel()


# ----------------------- DIMENSION REDUCERS FUNCTIONS -----------------------

def createDimensionReducer(reducer):
    if reducer == 'LDA':
        return LinearDiscriminantAnalysis(), reducer


def reduceDimensions(reducer, dataDescriptors):
    if reducer[1] == 'LDA':
        randomTagsAllSignTypes = randomMultiArray(dataDescriptors)
        mixedDataDescriptors = multiPickDataBySignType(dataDescriptors, randomTagsAllSignTypes)
        transformedData = reducer[0].fit_transform(mixedDataDescriptors, randomTagsAllSignTypes)
        return transformedData, randomTagsAllSignTypes


# ----------------------- LDA / BAYES FUNCTIONS -----------------------

def createLDAClassifiers():
    LDAClassifierProhibicionType = LinearDiscriminantAnalysis()
    LDAClassifierPeligroType = LinearDiscriminantAnalysis()
    LDAClassifierStopType = LinearDiscriminantAnalysis()
    LDAClassifierDirProhibidaType = LinearDiscriminantAnalysis()
    LDAClassifierCedaPasoType = LinearDiscriminantAnalysis()
    LDAClassifierDirObligatoriaType = LinearDiscriminantAnalysis()
    return LDAClassifierProhibicionType, LDAClassifierPeligroType, LDAClassifierStopType, LDAClassifierDirProhibidaType, LDAClassifierCedaPasoType, LDAClassifierDirObligatoriaType


def fitLDAClassifiers(LDAClassifiers, trainDataDescriptors):
    transformedDataBySignType = dict((signType, []) for signType in range(0, 6))
    negativeDataDescriptors = trainDataDescriptors[0]
    for signType in tqdm(range(1, 7)):
        randomTagsBySignType = randomBinaryArray(
            len(trainDataDescriptors[0]) + len(trainDataDescriptors[signType]),
            len(trainDataDescriptors[signType]), signType)
        mixedTrainDataDescriptors = pickDataBySignType(negativeDataDescriptors[:],
                                                       trainDataDescriptors[signType][:], randomTagsBySignType)
        transformedDataBySignType[signType - 1] = LDAClassifiers[signType - 1].fit_transform(
            mixedTrainDataDescriptors, randomTagsBySignType)
    sleep(0.02)


def predictProbabilityLDAClassifiers(LDAClassifiers, detectionsDescriptors, tolerance):
    probabilities = dict((signType, None) for signType in range(0, 6))

    onlyDescriptors, onlyRealSignTypes = extractDescriptorsAndRealSignTypes(detectionsDescriptors)

    print("\nCalculando probabilidades...")
    for signType in tqdm(range(0, 6)):
        probabilities[signType] = (LDAClassifiers[signType].predict_proba(onlyDescriptors))
    sleep(0.02)

    print("\nCalculando mejor predicción...")
    predictedSignTypes = extractBestPredictions(probabilities, tolerance, len(onlyRealSignTypes))
    return predictedSignTypes, onlyRealSignTypes


# ----------------------- KNN FUNCTIONS -----------------------

def KNNClassifier(kNeighbors):
    return KNeighborsClassifier(n_neighbors=kNeighbors)


def fitKNNClassifier(KNNClassifier, dimensionReducer, trainDataDescriptors):
    dimensionReducedData, tags = reduceDimensions(dimensionReducer, trainDataDescriptors)
    KNNClassifier.fit(dimensionReducedData, tags)
    return dimensionReducer, dimensionReducedData, tags


def predictProbabilityKNNClassifiers(KNNClassifier, reducer, detectionDescriptors):
    onlyDescriptors, onlyRealSignTypes = extractDescriptorsAndRealSignTypes(detectionDescriptors)
    onlyReducedDescriptors = reducer[0].transform(onlyDescriptors)
    predictedSignTypes = KNNClassifier.predict(onlyReducedDescriptors)
    return predictedSignTypes, onlyRealSignTypes


# ----------------------- CLASSIFIERS FUNCTIONS -----------------------

def createClassifiers(classifier):
    if classifier == 'LDABAYES':
        print("\nSe creará un clasificador binario (señal vs no señal) por cada tipo de señal de "
              "tráfico...")
        return createLDAClassifiers(), classifier
    elif classifier == 'KNN':
        return KNNClassifier(4), classifier


def fitClassifiers(classifiers,reducer, trainDataDescriptorsOrderBySignType):
    if classifiers[1] == 'LDABAYES':
        fitLDAClassifiers(classifiers[0], trainDataDescriptorsOrderBySignType)
        return None, None, None
    elif classifiers[1] == 'KNN':
        return tqdm(fitKNNClassifier(classifiers[0], createDimensionReducer(reducer),
                                     trainDataDescriptorsOrderBySignType))


def predictProbability(classifiers, reducer, testDataDescriptors, tolerance):
    if classifiers[1] == 'LDABAYES':
        return predictProbabilityLDAClassifiers(classifiers[0], testDataDescriptors, tolerance)
    elif classifiers[1] == 'KNN':
        print("\nCalculando probabilidades...")
        return tqdm(predictProbabilityKNNClassifiers(classifiers[0], reducer, testDataDescriptors))


def extractBestPredictions(probabilities, tolerance, numInstances):
    bestPredictions = []
    for prediction in tqdm(range(numInstances)):
        bestInstancePredictions = []
        for classifier in range(0, 6):
            noSignProb = probabilities[classifier][prediction][0]
            signProb = probabilities[classifier][prediction][1]
            bestInstancePredictions.append((max(noSignProb, signProb), 0 if noSignProb > signProb else classifier + 1))
        if noSignDetected(bestInstancePredictions, tolerance):
            bestPredictions.append(0)
        else:
            bestPred = max(bestInstancePredictions, key=lambda x: x[0] if x[1] != 0 else -math.inf)
            bestPredictions.append(bestPred[1])
    sleep(0.02)
    return bestPredictions


# ----------------------- VALIDATION TEST -----------------------

def testValidation(trainPath, mserParams, classifierParams, validationPercentage, toleranceNoSignal):
    constants.TRAIN_PATH = trainPath
    constants.TRAIN_PATH_REAL_RESULTS = trainPath + "/gt.txt"

    print("\nVa a comenzar el test de validación de detección de señales de tráfico mediante " + str(
        classifierParams[0]) + "-" + str(classifierParams[1]) + "-" + str(classifierParams[2]) + "...")
    print("\nInicializando detector MSER...")
    try:
        mser = initializeMSER(mserParams)
    except Exception as e:
        print("Ha ocurrido un error generando el detector :(")
        print("\n"
              "------------------------------------------------------------\n"
              "                        TEST FALLIDO                        \n"
              "------------------------------------------------------------\n")
        print(e)
    else:
        print("Se ha creado con éxito el detector MSER con parámetros:\n")
        print("   DELTA:", mserParams[0])
        print("   MIN AREA:", mserParams[1])
        print("   MAX AREA:", mserParams[2])
        print("   MAX VARIATION:", mserParams[3])

        print("\nInicializando descriptor de características " + str(classifierParams[0]) + "...")
        try:
            initializedDescriptor = initializeFeatureDescriptor(classifierParams[0])
        except Exception as e:
            print("Ha ocurrido un error generando el descriptor :(")
            print("\n"
                  "------------------------------------------------------------\n"
                  "                        TEST FALLIDO                        \n"
                  "------------------------------------------------------------\n")
            print(e)
        else:
            print("Se ha creado con éxito el descriptor " + str(classifierParams[0]) + " con parámetros:\n" if classifierParams[0] == 'HOG' else "Se ha creado con éxito el descriptor " + str(classifierParams[0]))
            if classifierParams[0] == 'HOG':
                for value, name in zip(constants.HOG_FEATURE_DESCRIPTORS_PARAMS[0], constants.HOG_FEATURE_DESCRIPTORS_PARAMS[1]):
                    print("   " + str(name) + ":", value)

            print("\nIniciando carga de datos de entrenamiento...")
            try:
                trainDataOrderBySignType, trainImages = loadTrainData(mser)
            except Exception as e:
                print("Ha ocurrido un error cargando los datos de entrenamiento :(")
                print("\n"
                      "------------------------------------------------------------\n"
                      "                        TEST FALLIDO                        \n"
                      "------------------------------------------------------------\n")
                print(e)
            else:
                try:
                    print("\nIniciando extracción de datos para validación a partir de datos de prueba (se "
                          "seleccionará un " + str(validationPercentage * 100) + "% de los datos de entrenamiento para "
                                                                           "validar)...")
                    trainDataOrderBySignType, testDataOrderBySignType = extractEvaluationTestResults(
                        trainDataOrderBySignType, validationPercentage)
                except Exception as e:
                    print("Ha ocurrido un error extrayendo los datos de validación :(")
                    print("\n"
                          "------------------------------------------------------------\n"
                          "                        TEST FALLIDO                        \n"
                          "------------------------------------------------------------\n")
                    print(e)
                else:
                    try:
                        print("\nExtrayendo vectores de características " + str(
                            classifierParams[0]) + " de los datos de prueba...")
                        trainDataDescriptorsOrderBySignType = calculateDescriptors(trainDataOrderBySignType,
                                                                                   initializedDescriptor)
                        print("\nExtrayendo vectores de características " + str(
                            classifierParams[0]) + " de los datos de validación...")
                        testDataDescriptorsOrderBySignType = calculateDescriptors(testDataOrderBySignType,
                                                                                  initializedDescriptor)
                    except Exception as e:
                        print("Ha ocurrido un error extrayendo los vectores de características " + str(
                            classifierParams[0]) + " :(")
                        print("\n"
                              "------------------------------------------------------------\n"
                              "                        TEST FALLIDO                        \n"
                              "------------------------------------------------------------\n")
                        print(e)
                    else:
                        try:
                            print("\nCreación de los clasificadores " + str(classifierParams[2]) + "...")
                            classifiers = createClassifiers(classifierParams[2])
                        except Exception as e:
                            print("Ha ocurrido un error creando los clasificadores " + str(classifierParams[2]) + " :(")
                            print("\n"
                                  "------------------------------------------------------------\n"
                                  "                        TEST FALLIDO                        \n"
                                  "------------------------------------------------------------\n")
                            print(e)
                        else:
                            try:
                                print("\nEntrenando los clasificadores " + str(
                                    classifierParams[2]) + " mediante los vectores de características "
                                                           "" + str(
                                    classifierParams[0]) + " de cada tipo de señal de tráfico...")
                                reducer, _, _ = fitClassifiers(classifiers, classifierParams[1], trainDataDescriptorsOrderBySignType)
                            except Exception as e:
                                print("Ha ocurrido un error entrenando los clasificadores " + str(
                                    classifierParams[2]) + " :(")
                                print("\n"
                                      "------------------------------------------------------------\n"
                                      "                        TEST FALLIDO                        \n"
                                      "------------------------------------------------------------\n")
                                print(e)
                            else:
                                try:
                                    print("\nRealizando predicciones mediante los datos de validación... (tolerancia "
                                          "para asumir no señal < " + str(toleranceNoSignal) + ")")
                                    print("\nAplicando formato a datos de validación...")
                                    testDataDescriptors = flatData(list(testDataDescriptorsOrderBySignType.values()))
                                    random.shuffle(testDataDescriptors)
                                    predictedSignTypes, trueSignTypes = predictProbability(classifiers, reducer,
                                                                                           testDataDescriptors, 0.5)
                                except Exception as e:
                                    print("Ha ocurrido un error realizando las predicciones :(")
                                    print("\n"
                                          "------------------------------------------------------------\n"
                                          "                        TEST FALLIDO                        \n"
                                          "------------------------------------------------------------\n")
                                    print(e)
                                else:
                                    print("\nEl proceso de predicción de tipo de señal de tráfico ha finalizado con "
                                          "éxito, a continuación se mostrarán los resultados...")
                                    try:
                                        print("\nMostrando matriz de confusión...")
                                        confusionMatrix = confusion_matrix(trueSignTypes, predictedSignTypes)
                                        matrixDisplay = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,
                                                                               display_labels=constants.SIGN_NAMES)
                                        matrixDisplay.plot(cmap='Blues', xticks_rotation=45)
                                        plt.title("clasificador " + str(classifierParams[0]) + "-" + str(
                                            classifierParams[1]) + "-" + str(classifierParams[2]))
                                        plt.tight_layout()
                                        plt.show()
                                    except Exception as e:
                                        print("Ha ocurrido un error mostrando la matriz de confusión :(")
                                        print("\n"
                                              "------------------------------------------------------------\n"
                                              "                        TEST FALLIDO                        \n"
                                              "------------------------------------------------------------\n")
                                        print(e)
                                    else:
                                        try:
                                            print("\nMostrando reporte de resultados del clasificador " + str(
                                                classifierParams[0]) + "-" + str(classifierParams[1]) + "-" + str(
                                                classifierParams[2]) + "...")
                                            classificationReport = classification_report(trueSignTypes,
                                                                                         predictedSignTypes,
                                                                                         target_names=constants.SIGN_NAMES)
                                            print(classificationReport)
                                        except Exception as e:
                                            print("Ha ocurrido un error mostrando el reporte de resultados :(")
                                            print("\n"
                                                  "------------------------------------------------------------\n"
                                                  "                        TEST FALLIDO                        \n"
                                                  "------------------------------------------------------------\n")
                                            print(e)
                                        else:
                                            print("\n"
                                                  "------------------------------------------------------------\n"
                                                  "                  TEST TERMINADO CON ÉXITO                  \n"
                                                  "------------------------------------------------------------\n")
