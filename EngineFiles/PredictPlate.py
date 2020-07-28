import cv2
import numpy as np

class PredictPlate:
    def __init__(self):
        self.iPlate = None
        self.iGrayScale = None
        self.iThresh = None
        self.locPlate = None
        self.sChars = ""