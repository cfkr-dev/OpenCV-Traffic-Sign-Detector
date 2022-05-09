import argparse
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tempfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    # Load training data

    # Create the detector

    # Load testing data

    # Evaluate sign detections


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

        return x1, y1, x2, y2
    else:
        return None


def drawWindowsOnImage(windowsCoords, image):
    x1, y1, x2, y2 = windowsCoords
    return cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)


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


def checkIfImageIsDuplicatedOrMergeSimilarOnes(image, detections, tolerance):
    if detections:
        lastDetection = detections[-1]

        """ Use histogram comparison between two images || resource from: 
        https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html """

        similarity = cv2.compareHist(calculateHistAndNormalize(image), calculateHistAndNormalize(lastDetection), cv2.HISTCMP_CORREL)

        if similarity > tolerance:
            return None, True, False
        elif 0.50 <= similarity <= tolerance:
            image = cv2.addWeighted(image, 0.5, lastDetection, 0.5, 0.0)
            return image, False, True
        else:
            return image, False, False
    else:
        return image, False, False


def cleanDuplicatedDetections(imageDetections):
    cleanDetections = []

    for image in imageDetections:
        image, duplicated, merged = checkIfImageIsDuplicatedOrMergeSimilarOnes(image, cleanDetections, 0.85)
        if not duplicated:
            if merged:
                cleanDetections.pop()

            cleanDetections.append(image)

            plt.imshow(image)
            plt.show()

    return cleanDetections


def MSERTrafficSignDetector(image):
    modifiedImage = grayAndEnhanceContrast(image)
    # showImage('Original image', modifiedImage)

    mser = cv2.MSER_create(delta=5, min_area=200, max_area=2000, max_variation=0.1)
    windowsBorders = mser.detectRegions(modifiedImage)[1]

    croppedImageDetections = []
    i = 0
    for window in windowsBorders:

        i += 1
        print(i)

        if i == 55:
            print('error')

        windowCords = makeWindowBiggerOrDiscardFakeDetections(window, 1.25)
        if windowCords is not None:
            croppedImageDetections.append(cv2.resize(cropImageByCoords(windowCords, image), (25, 25)))

    croppedImageDetections = cleanDuplicatedDetections(croppedImageDetections)

    return croppedImageDetections


def grayAndEnhanceContrast(image):
    # Img turn gray
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurImage = cv2.GaussianBlur(grayImage, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(1, 1))
    claheImage = clahe.apply(blurImage)
    contrastAndBrightnessCorrectionImage = cv2.convertScaleAbs(claheImage, alpha=3, beta=-500)
    # threshImage = cv2.adaptiveThreshold(new_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 29, -4)
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

    for signType in signs:
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


def showImage(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    path = 'test_alumnos_jpg'
    files = os.listdir(path)
    for file in files:
        if not file.endswith('.txt'):
            print(file)
        detections = MSERTrafficSignDetector(cv2.imread(path + '/' + file))

        for detection in detections:
            showImage('detección', detection)


# path = 'test_alumnos_jpg'
# files = os.listdir(path)
# for file in files:
#     if not file.endswith('.txt'):
#         print(file)
#         main(path + '/' + file)

# calculateMeanMask()

main()
