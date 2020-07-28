import cv2
import numpy as np
import math

class PredictChar:
    def __init__(self, contour):
        self.contour = contour
        self.bRect = cv2.boundingRect(self.contour)
        [x, y, w, h] = self.bRect
        self.bRectX = x
        self.bRectY = y
        self.bRectW = w
        self.bRectH = h
        self.bRectArea = self.bRectW * self.bRectH
        self.centerX = (self.bRectX * 2 + self.bRectW)/2
        self.centerY = (self.bRectY * 2 + self.bRectH)/2
        self.FlatDiagS = math.sqrt((self.bRectW**2) + (self.bRectH**2))
        self.FlatAspRat = float(self.bRectW) / float(self.bRectH)