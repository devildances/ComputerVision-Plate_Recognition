import cv2
import numpy as np
import math

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

def extracting(img):
    h, w, numC = img.shape
    hsv = np.zeros((h,w,3), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, val = cv2.split(hsv)
    return val

def MaxCont(img):
    h, w = img.shape
    topH = np.zeros((h, w, 1), np.uint8)
    blackH = np.zeros((h, w, 1), np.uint8)
    structEl = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    topH = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, structEl)
    blackH = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, structEl)
    GrayScaleAddTopH = cv2.add(img, topH)
    GrayScaleAddTopHReduceBlackH = cv2.subtract(GrayScaleAddTopH, blackH)

    return GrayScaleAddTopHReduceBlackH

def preprocessing(img):
    GrayScale = extracting(img)
    MaxContGrayScale = MaxCont(GrayScale)
    h, w = GrayScale.shape
    blurring = np.zeros((h, w, 1), np.uint8)
    blurring = cv2.GaussianBlur(MaxContGrayScale,GAUSSIAN_SMOOTH_FILTER_SIZE,0)
    thresh = cv2.adaptiveThreshold(blurring,
                                    255.0,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    ADAPTIVE_THRESH_BLOCK_SIZE,
                                    ADAPTIVE_THRESH_WEIGHT)
    return GrayScale, thresh