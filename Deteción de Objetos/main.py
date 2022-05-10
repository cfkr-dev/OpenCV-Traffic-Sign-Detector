import argparse
import os
import shutil
from time import sleep

import cv2
import numpy as np
from matplotlib import pyplot as plt
import tempfile

# Loop progress bar || Resource from: https://github.com/tqdm/tqdm
from tqdm import tqdm

# Use GPU to improve performance || Resource form: https://www.geeksforgeeks.org/running-python-script-on-gpu/
from numba import jit, cuda


def makeWindowBiggerOrDiscardFakeDetections(window, percentage):
    x1, y1, w, h = window

    x2 = x1 + w
    y2 = y1 + h

    middleDeltaW = w * (percentage - 1) * 0.5
    middleDeltaH = h * (percentage - 1) * 0.5

    squareGoodAspectRatio = True if (0.8 < w / h < 1.2) else False
    if squareGoodAspectRatio:

        x1 = x1 - middleDeltaW if x1 - middleDeltaW > 0 else 0
        y1 = y1 - middleDeltaH if y1 - middleDeltaH > 0 else 0
        x2 = x2 + middleDeltaW if x2 + middleDeltaW > 0 else 0
        y2 = y2 + middleDeltaH if y2 + middleDeltaH > 0 else 0

        return int(x1), int(y1), int(x2), int(y2)
    else:
        return None


def cropImageByCoords(coords, image):
    x1, y1, x2, y2 = np.asarray(coords).astype(int)
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


# @cuda.jit(target="cuda")
def checkIfImageIsDuplicatedOrMergeSimilarOnes(image, detections, tolerance):
    deletions = []
    if detections:
        for detection in detections:

            """ Use histogram comparison between two images || resource from: 
                    https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html """
            similarity = cv2.compareHist(calculateHistAndNormalize(image), calculateHistAndNormalize(detection),
                                         cv2.HISTCMP_CORREL)

            if similarity > tolerance:
                deletions.append(detection)
            elif 0.45 <= similarity <= tolerance:
                image = cv2.addWeighted(image, 0.5, detection, 0.5, 0.0)
                deletions.append(detection)

    return image, deletions


# @cuda.jit(target="cuda")
def getElementIndexFromList(l, element):
    # Consider "l" contains "element"
    index = 0
    for x in l:
        if np.array_equal(x, element):
            return index
        index += 1


# @cuda.jit(target="cuda")
def cleanDuplicatedDetections(imageDetections):
    cleanDetections = []

    for image in imageDetections:
        image, deletions = checkIfImageIsDuplicatedOrMergeSimilarOnes(image, cleanDetections, 0.85)
        if deletions:
            for deletedImage in deletions:
                cleanDetections.pop(getElementIndexFromList(cleanDetections, deletedImage))

        cleanDetections.append(image)

    return cleanDetections


def createImageWithWindows(image, windowsBorders):
    for x1, y1, x2, y2 in windowsBorders:
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return image


def detectSignsOnDirectory(path, mser):
    directoryDetections = []
    numberOfDetections = []
    imagesWithWindows = []
    for file in tqdm(os.listdir(path)):
        if not file.endswith('.txt'):
            detections, windowsBorders = MSERTrafficSignDetector(cv2.imread(path + '/' + file), mser)
            directoryDetections.append(detections)
            numberOfDetections.append((file, len(detections)))
            imagesWithWindows.append((file, createImageWithWindows(cv2.imread(path + '/' + file), windowsBorders)))
    sleep(0.02)
    return directoryDetections, numberOfDetections, imagesWithWindows


# @cuda.jit(target="cuda")
def MSERTrafficSignDetector(image, mser):
    modifiedImage = grayAndEnhanceContrast(image)

    windowsBorders = mser.detectRegions(modifiedImage)[1]

    croppedImageDetections = []
    windowsBordersWithoutFakeDetections = []
    for window in windowsBorders:

        windowCords = makeWindowBiggerOrDiscardFakeDetections(window, 1.30)
        if windowCords is not None:
            croppedImageDetections.append(cv2.resize(cropImageByCoords(windowCords, image), (25, 25)))
            windowsBordersWithoutFakeDetections.append(windowCords)

    croppedImageDetections = cleanDuplicatedDetections(croppedImageDetections)

    return croppedImageDetections, windowsBordersWithoutFakeDetections


def grayAndEnhanceContrast(image):
    # Img turn gray
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurImage = cv2.GaussianBlur(grayImage, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(1, 1))
    claheImage = clahe.apply(blurImage)
    contrastAndBrightnessCorrectionImage = cv2.convertScaleAbs(claheImage, alpha=3, beta=-500)
    return contrastAndBrightnessCorrectionImage


def HSVAzulRojo(image, color):
    imageResize = cv2.resize(image, (25, 25))
    imageHSV = cv2.cvtColor(imageResize, cv2.COLOR_BGR2HSV)

    # Red color
    if color == 'r':
        # Lower red mask
        lowerRed = np.array([0, 50, 50])
        upperRed = np.array([10, 255, 255])
        maskLower = cv2.inRange(imageHSV, lowerRed, upperRed)

        # Upper red mask
        lowerRed = np.array([170, 50, 50])
        upperRed = np.array([180, 255, 255])
        maskUpper = cv2.inRange(imageHSV, lowerRed, upperRed)

        maskRed = cv2.add(maskLower, maskUpper)

        return maskRed

    # Blue color = b
    elif color == 'b':
        lowerBlue = np.array([90, 70, 50], np.uint8)
        upperBlue = np.array([128, 255, 255], np.uint8)
        maskBlue = cv2.inRange(imageHSV, lowerBlue, upperBlue)

        return maskBlue


def calculateMeanMask():
    tempdir = tempfile.mkdtemp(prefix="meanMasks-")

    prohibicion = ['00', '01', '02', '03', '04', '05', '07', '08', '09', '10', '15', '16']
    peligro = ['11', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
    stop = ['14']
    direccionProhibida = ['17']
    cedaPaso = ['13']
    direccionObligatoria = ['38']

    signs = [prohibicion, peligro, stop, direccionProhibida, cedaPaso, direccionObligatoria]

    namesList = ['prohibicion', 'peligro', 'stop', 'direccionProhibida', 'cedaPaso', 'direccionObligatoria']
    namesListPosition = -1

    for signType in tqdm(signs):
        namesListPosition += 1
        mask = np.zeros((25, 25, 3), np.uint8)
        first = True
        for singDir in signType:
            signs = os.listdir('train_jpg/' + singDir)
            for file in signs:
                currentResizedImage = cv2.resize(cv2.imread('train_jpg/' + singDir + '/' + file), (25, 25))
                if first:
                    mask = cv2.addWeighted(currentResizedImage, 1, mask, 0, 0.0)
                    first = False
                else:
                    mask = cv2.addWeighted(currentResizedImage, 0.5, mask, 0.5, 0.0)

        if namesListPosition == 5:
            cv2.imwrite(tempdir + '/' + namesList[namesListPosition] + '.jpg', HSVAzulRojo(mask, 'b'))
        else:
            cv2.imwrite(tempdir + '/' + namesList[namesListPosition] + '.jpg', HSVAzulRojo(mask, 'r'))
        sleep(0.01)
    return tempdir


def showImage(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test(trainPath, testPath):
    print("\nVa a comenzar el test de detección de señales de tráfico...")
    print("\nGenerando mascaras a partir de imágenes de entrenamiento... (", trainPath, ")")

    try:
        masksDir = calculateMeanMask()
    except Exception as e:
        print("Ha ocurrido un problema generando las máscaras :(   (", e, ")")
    else:
        print("Máscarás generadas con éxito en", masksDir)

    print("\nIniciando detector MSER...")

    mser = None

    try:
        delta = 5
        minArea = 200
        maxArea = 2000
        maxVariation = 0.1

        mser = cv2.MSER_create(delta=delta, min_area=minArea, max_area=maxArea, max_variation=maxVariation)
    except Exception as e:
        print("Ha ocurrido un error generando el detector :(   (", e, ")")
    else:
        print("Se ha creado con éxito el detector MSER con parámetros:\n")
        print("   DELTA:", delta)
        print("   MIN AREA:", minArea)
        print("   MAX AREA:", maxArea)
        print("   MAX VARIATION:", maxVariation)
    finally:

        print("\nVa a comenzar la detección de señales de tráfico en las imágenes de test... (", testPath, ")\n")
        print("Analizando y extrayendo regiones de interés...")

        try:
            detections, numDetections, imagesWithDetections = detectSignsOnDirectory(testPath, mser)
        except Exception as e:
            print("Ha ocurrido un error en el proceso de detección de señales :(   (", e, ")")
        else:
            resultImagesPath = "resultado_imgs"

            print("\nEl proceso ha concluido con éxito, las imágenes de test con sus respectivas detecciones (sin "
                  "eliminación de repeticiones) serán "
                  "guardadas en", resultImagesPath)

            print("\nGenerando resultados...")

            if os.path.isdir(resultImagesPath):
                shutil.rmtree(resultImagesPath)

            os.mkdir(resultImagesPath)

            for file, image in tqdm(imagesWithDetections):
                cv2.imwrite(resultImagesPath + "/" + file, image)
                sleep(0.02)

            print(
                "\nA continuación se listarán las detecciones obtenidas, con eliminación de repeticiones, por cada "
                "archivo en",
                testPath, "\n")

            total = 0
            for file, number in numDetections:
                print(file, ".......", number, "   detecciones" if number < 10 else "  detecciones")
                total += number
            print("Total ...........", total, "detecciones")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description='Trains and executes a given detector over a set of testing images')
#     parser.add_argument(
#         '--detector', type=str, nargs="?", default="", help='Detector string name')
#     parser.add_argument(
#         '--train_path', default="", help='Select the training data dir')
#     parser.add_argument(
#         '--test_path', default="", help='Select the testing data dir')
#
#     args = parser.parse_args()
#
#     test()

test("train_jpg", "test_alumnos_jpg")
