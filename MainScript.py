import cv2
import numpy as np
import os

import EngineFiles.ReadChars as RC
import EngineFiles.ReadPlates as RP

'''
BGR Color Palettes
B-G-R formatting
'''
BGR_black = (0.0, 0.0, 0.0)
BGR_white = (255.0, 255.0, 255.0)
BGR_yellow = (0.0, 255.0, 255.0)
BGR_green = (0.0, 255.0, 0.0)
BGR_red = (0.0, 0.0, 255.0)

showSteps = False

def PlateRecognition():
    KNNmodelTrain = RC.KNNmodeling()
    if KNNmodelTrain == False:
        print("\nKNN modeling and training was not completed and raised error!")
        return None

    licPlateImg = cv2.imread('LicPlates/1.png')

    if licPlateImg is None:
        print("\nimage not found or the file is corrupted!")
        os.system('pause')
        return None

    ListPlates = RP.CropPlates(licPlateImg)
    ListPlates = RC.GetCharsInPlate(ListPlates)

    cv2.imshow('Vehicle Plate Image', licPlateImg)

# =================================================================================

if __name__ == '__main__':
    PlateRecognition()