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

    squareGoodAspectRatio = True if (0.8 < w / h < 1.20) else False
    if squareGoodAspectRatio:

        x1 = x1 - middleDeltaW if x1 - middleDeltaW > 0 else 0
        y1 = y1 - middleDeltaH if y1 - middleDeltaH > 0 else 0
        x2 = x2 + middleDeltaW if x2 + middleDeltaW > 0 else 0
        y2 = y2 + middleDeltaH if y2 + middleDeltaH > 0 else 0

        return int(x1), int(y1), int(x2), int(y2)
    else:
        return None


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


# @cuda.jit(target="cuda")
def meanCoods(coordsA, coordsB):
    x1A, y1A, x2A, y2A = coordsA
    x1B, y1B, x2B, y2B = coordsB
    return (x1A + x1B) // 2, (y1A + y1B) // 2, (x2A + x2B) // 2, (y2A + y2B) // 2


def checkIfImageIsDuplicatedOrMergeSimilarOnes(image, detections, tolerance):
    deletions = []
    if detections:
        for detection in detections:

            """ Use histogram comparison between two images || resource from: 
                    https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html """
            similarity = cv2.compareHist(calculateHistAndNormalize(image[0]), calculateHistAndNormalize(detection[0]),
                                         cv2.HISTCMP_CORREL)

            if similarity > tolerance:
                deletions.append(detection)
            elif 0.75 <= similarity <= tolerance:
                image = (cv2.addWeighted(image[0], 0.5, detection[0], 0.5, 0.0), meanCoods(image[1], detection[1]))
                deletions.append(detection)

    return image, deletions


# @cuda.jit(target="cuda")
def getElementIndexFromList(l, element):
    # Consider "l" contains "element"
    index = 0
    for x in l:
        if np.array_equal(x[0], element):
            return index
        index += 1


# @cuda.jit(target="cuda")
def cleanDuplicatedDetections(imageDetections):
    cleanDetections = []

    for image in imageDetections:
        image, deletions = checkIfImageIsDuplicatedOrMergeSimilarOnes(image, cleanDetections, 0.85)
        if deletions:
            for deletedImage in deletions:
                cleanDetections.pop(getElementIndexFromList(cleanDetections, deletedImage[0]))

        cleanDetections.append(image)

    return cleanDetections


def createImageWithWindows(image, windowsBorders):
    for detectedImage in windowsBorders:
        x1, y1, x2, y2 = detectedImage[1]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return image


def detectSignsOnDirectory(path, mser):
    directoryDetections = []
    numberOfDetections = []
    imagesWithWindows = []
    for file in tqdm(os.listdir(path)):
        if not file.endswith('.txt'):
            detections = MSERTrafficSignDetector(cv2.imread(path + '/' + file), mser)
            directoryDetections.append(detections)
            numberOfDetections.append((file, len(detections)))
            image = createImageWithWindows(cv2.imread(path + '/' + file), detections)
            imagesWithWindows.append((file, image))

    sleep(0.02)
    return directoryDetections, numberOfDetections, imagesWithWindows


# @cuda.jit(target="cuda")
def MSERTrafficSignDetector(image, mser):
    modifiedImage = grayAndEnhanceContrast(image)

    windowsBorders = mser.detectRegions(modifiedImage)[1]

    croppedImageDetections = []
    for window in windowsBorders:

        windowCords = makeWindowBiggerOrDiscardFakeDetections(window, 1.30)
        if windowCords is not None:
            croppedImageDetections.append((cv2.resize(cropImageByCoords(windowCords, image), (25, 25)), windowCords))

    croppedImageDetections = cleanDuplicatedDetections(croppedImageDetections)

    return croppedImageDetections


# Gamma correction for enhance exposure || Resource from:
# https://lindevs.com/apply-gamma-correction-to-an-image-using-opencv/

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def grayAndEnhanceContrast(image):
    # Img turn gray

    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2)
    claheImage = clahe.apply(grayImage)
    blurImage = cv2.GaussianBlur(claheImage, (3, 3), 0)
    imageGammaCorrection = gammaCorrection(blurImage, 2)

    result = imageGammaCorrection

    return result


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
    return tempdir  # Acordase de eliminar al terminar la ejecución el dir temporal


# def showImage(title, image):
#     cv2.imshow(title, image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


def detectionsMaskCorrelation(detections, masksDir):
    raise TypeError("No está definido el comportamiento de esta función")


# detections -> (str filename; int x1; int y1; int x2; int y2; int signType; float score)
def createDetectionsStrings(detections):
    detectionsStrings = []
    for detection in detections:
        filename, x1, y1, x2, y2, signType, score = detection
        detectionsStrings.append(
            filename.split(".", 1)[0] + ".ppm;" + x1 + ";" + y1 + ";" + x2 + ";" + y2 + ";" + signType + ";" + score)
    return detectionsStrings


def calculateSignType(signType):
    prohibicion = ['00', '01', '02', '03', '04', '05', '07', '08', '09', '10', '15', '16']
    peligro = ['11', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
    stop = ['14']
    direccionProhibida = ['17']
    cedaPaso = ['13']
    direccionObligatoria = ['38']

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

# generateStatistics -> (list detectionsPerFileByType[str fileName,
# list detectionsByTypeOnFile[(str type , int totalCorrectByTypeOnFile ,
# int totalIncorrectByTypeOnFile, int expectedTotalCorrectByTypeOnFile) ,
# int totalCorrectOnFile , int totalIncorrectOnFile, int expectedTotalCorrectOnFile]]
# ; list totalDetectionsByType[(str type , int totalCorrectByType ,
# int totalIncorrectByType , int expectedTotalCorrectByType)]; int totalCorrect ; int
# totalIncorrect ; expectedTotalCorrect)


# AÑADIR DUPLICADOS -> NO CONTABILIZARLOS NI EN ACIERTOS NI EN FALLOS

# detections -> (str filename; int x1; int y1; int x2; int y2; int signType; float score)

def generateStatistics(detections, realResultsFilePath, fileNames):
    realResults = []
    file = open(realResultsFilePath, "r")
    for line in file:

        filename, x1, y1, x2, y2, signType = line.split(';')

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        signType = calculateSignType(signType)

        realResults.append((filename, x1, y1, x2, y2, signType))

    for fileName in fileNames:
        detectionsOnFile, realResultsOnFile = getResultsOnFile(fileName[0], detections, realResults)







    raise TypeError("No está definido el comportamiento de esta función")


def test(trainPath, testPath, MSERValues):
    print("\nVa a comenzar el test de detección de señales de tráfico...")
    print("\nGenerando mascaras a partir de imágenes de entrenamiento... \n(" + trainPath + ")")

    try:
        masksDir = calculateMeanMask()
    except Exception as e:
        print("Ha ocurrido un problema generando las máscaras :(")
        print("\n"
              "------------------------------------------------------------\n"
              "                        TEST FALLIDO                        \n"
              "------------------------------------------------------------\n")
        print(e)
    else:
        print("Máscaras generadas con éxito en", masksDir)

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
                "\nVa a comenzar la detección de señales de tráfico en las imágenes de test... \n(" + testPath + ")\n")
            print("Analizando y extrayendo regiones de interés...")

            try:
                detections, numDetections, imagesWithDetections = detectSignsOnDirectory(testPath, mser)
            except Exception as e:
                print("Ha ocurrido un error en el proceso de detección de señales :(")
                print("\n"
                      "------------------------------------------------------------\n"
                      "                        TEST FALLIDO                        \n"
                      "------------------------------------------------------------\n")
                print(e)
            else:
                resultImagesPath = "resultado_imgs"  # _d" + str(delta) + "_mv" + str(maxVar)  # Borrar después de testing

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
                        testPath + "\n")

                    total = 0
                    for file, number in numDetections:
                        print(file, ".......", number, "   detecciones" if number < 10 else "  detecciones")
                        total += number
                    print("Total ...........", total, "detecciones")

                    print("\nVa a comenzar el proceso de filtrado por correlación de máscaras almacenadas en el "
                          "directorio"
                          , trainPath + "...\n")
                    print("Realizando el filtrado...")
                    try:
                        # detections -> (str filename; int x1; int y1; int x2; int y2; int signType; float score)
                        detections = detectionsMaskCorrelation(detections, masksDir)
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
                        print("nombre_archivo.ppm;x1_coord;y1_coord;x2_coord;y2_coord;tipo_señal;score\n")
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
                            for detection in detectionsStrings:
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

                                # generateStatistics -> (list detectionsPerFileByType[str fileName,
                                # list detectionsByTypeOnFile[(str type , int totalCorrectByTypeOnFile ,
                                # int totalIncorrectByTypeOnFile, int expectedTotalCorrectByTypeOnFile) ,
                                # int totalCorrectOnFile , int totalIncorrectOnFile, int expectedTotalCorrectOnFile]]
                                # ; list totalDetectionsByType[(str type , int totalCorrectByType ,
                                # int totalIncorrectByType , int expectedTotalCorrectByType)]; int totalCorrect ; int
                                # totalIncorrect ; expectedTotalCorrect)

                                # AÑADIR DUPLICADOS -> NO CONTABILIZARLOS NI EN ACIERTOS NI EN FALLOS

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

                                detectionsPerFileByType, totalDetectionsByType, totalCorrect, totalIncorrect, expectedTotalCorrect = statisticsResults

                                print("\n"
                                      "---------------------------------\n"
                                      "     DETECCIONES POR ARCHIVO     \n"
                                      "---------------------------------\n")
                                for detectionPerFile in detectionsPerFileByType:
                                    fileName, detectionsByTypeOnFile, totalCorrectOnFile, totalIncorrectOnFile, expectedTotalCorrectOnFile = detectionPerFile
                                    print(fileName, "...............................")
                                    print("                TOTAL CORRECTAS:", totalCorrectOnFile)
                                    print("              TOTAL INCORRECTAS:", totalIncorrectOnFile)
                                    print("        VALOR CORRECTO ESPERADO:", expectedTotalCorrectOnFile)
                                    print("                TASA DE ACIERTO:",
                                          totalCorrectOnFile / expectedTotalCorrectOnFile)
                                    for detectionByTypeOnFile in detectionsByTypeOnFile:
                                        signType, totalCorrectByTypeOnFile, totalIncorrectByTypeOnFile, expectedTotalCorrectByTypeOnFile = detectionByTypeOnFile
                                        print("\n            " + signType + ":..................................")
                                        print("                                  TOTAL CORRECTAS:",
                                              totalCorrectByTypeOnFile)
                                        print("                                TOTAL INCORRECTAS:",
                                              totalIncorrectByTypeOnFile)
                                        print("                          VALOR CORRECTO ESPERADO:",
                                              expectedTotalCorrectByTypeOnFile)
                                        print("                                  TASA DE ACIERTO:",
                                              totalCorrectByTypeOnFile / expectedTotalCorrectByTypeOnFile)

                                print("\n"
                                      "---------------------------------------\n"
                                      "     DETECCIONES POR TIPO DE SEÑAL     \n"
                                      "---------------------------------------\n")
                                for detectionByType in totalDetectionsByType:
                                    signType, totalCorrectByType, totalIncorrectByType, expectedTotalCorrectByType = detectionByType
                                    print(signType, "...............................")
                                    print("                TOTAL CORRECTAS:", totalCorrectByType)
                                    print("              TOTAL INCORRECTAS:", totalIncorrectByType)
                                    print("        VALOR CORRECTO ESPERADO:", expectedTotalCorrectByType)
                                    print("                TASA DE ACIERTO:",
                                          totalCorrectByType / expectedTotalCorrectByType)

                                print("\n"
                                      "-----------------------------\n"
                                      "     DETECCIONES TOTALES     \n"
                                      "-----------------------------\n")
                                print("TOTAL CORRECTAS:", totalCorrect)
                                print("TOTAL INCORRECTAS:", totalIncorrect)
                                print("VALOR CORRECTO ESPERADO:", expectedTotalCorrect)
                                print("TASA DE ACIERTO:", totalCorrect / expectedTotalCorrect)

                                print("\n"
                                      "------------------------------------------------------------\n"
                                      "                  TEST TERMINADO CON ÉXITO                  \n"
                                      "------------------------------------------------------------\n")

                                return totalCorrect / expectedTotalCorrect  # Borrar después de testing


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

def getTotal(e):
    return e[2]


def multiTest():
    delta = np.arange(7, 19, 1)
    variation = np.arange(0.05, 1.05, 0.05)

    totals = []
    for d in delta:
        for v in variation:
            total = test("train_jpg", "test_alumnos_jpg", (d, 200, 2000, v))
            totals.append((d, v, total))

    totals.sort(key=getTotal, reverse=True)
    for t in totals:
        print(t)

    print("\n\n\nVALOR ÓPTIMO!!!!!:", totals[0])


test("train_jpg", "test_alumnos_jpg", (25, 200, 2000, 1))

# multiTest()
