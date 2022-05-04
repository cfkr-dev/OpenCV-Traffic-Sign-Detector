import argparse
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

def equalizeImage(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(grayImage)


def showImage(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image = equalizeImage(cv2.imread('train_jpg/00/00000.jpg'))
    # showImage('Original image', image)

    mser = cv2.MSER_create(delta=5, max_variation=0.5, max_area=20000)
    detection, borders = mser.detectRegions(image)




main()


