import argparse
import os
from colorsys import hsv_to_rgb

import numpy
import cv2
import numpy as np
from numpy.random import random

from matplotlib import pyplot as plt

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


def grayAndEnhanceContrast(image):
    # Img turn gray
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Img Equalized
    equImage = cv2.equalizeHist(grayImage)
    # Use Contrast Limited Adaptive Histogram Equalization (Improve contrast)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(20, 20))
    claheImage = clahe.apply(equImage)
    # Reduce image noise
    noiselessImage = cv2.fastNlMeansDenoising(claheImage, None, 10, 7, 21)
    finalImage = noiselessImage

    return finalImage


def showImage(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(path):
    realImage = cv2.imread(path)
    modifiedImage = grayAndEnhanceContrast(cv2.imread(path))
    showImage('Original image', modifiedImage)

    mser = cv2.MSER_create(delta=3, min_area=200, max_area=800, max_variation=0.2)
    detection, borders = mser.detectRegions(modifiedImage)

    for box in borders:
        x, y, w, h = box
        squareGoodAspectRatio = True if (0.8 < w / h < 1.2) else False
        if squareGoodAspectRatio:
            cv2.rectangle(realImage, (x, y), (x + w, y + h), (255, 0, 0), 1)
    plt.imshow(realImage)
    plt.show()


path = 'test_alumnos_jpg'
files = os.listdir(path)
for file in files:
    if not file.endswith('.txt'):
        print(file)
        main(path + '/' + file)
