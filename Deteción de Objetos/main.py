import argparse
import os
import cv2
import numpy as np


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

    # Blue color = 2
    lowerBlue = np.array([90, 70, 50], np.uint8)
    upperBlue = np.array([128, 255, 255], np.uint8)
    maskBlue = cv2.inRange(imageHSV, lowerBlue, upperBlue)

    return maskBlue

def showImage(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(path):
    realImage = cv2.imread(path)

    imageRed = HSVAzulRojo(realImage, 'r')
    imageBlue = HSVAzulRojo(realImage, 'b')

    showImage('Imagen roja', imageRed)
    showImage('Imagen azul', imageBlue)

    path = 'test_alumnos_jpg'
    files = os.listdir(path)
    for file in files:
        if not file.endswith('.txt'):
            print(file)
            main(path + '/' + file)
