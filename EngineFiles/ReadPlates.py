import cv2
import numpy as np
import math
import random

import MainScript
import EngineFiles.ReadChars as RC
import EngineFiles.Preprocessing as PP
import EngineFiles.PredictChar as PC
import EngineFiles.PredictPlate as PPL

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


def lookupChars(img1):
    ListChars, countChars, imgC = [], 0, img1.copy()
    contours, hierC = cv2.findContours(imgC, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img1.shape
    img1Cont = np.zeros((h,w,3), np.uint8)

    for i in range(len(contours)):
        if MainScript.showSteps == True:
            cv2.drawContours(img1Cont, contours, i, MainScript.BGR_white,)

        PredictChar = PC.PredictChar(contours[i])

        if RC.checkChar(PredictChar):
            countChars += 1
            ListChars.append(PredictChar)

    if MainScript.showSteps == True:
        print('\nstep 2 - len(contours) =',len(contours))
        print('\nstep 2 - countChars =', countChars)
        cv2.imshow('2a',img1Cont)

    return ListChars


def gainPlate(imgO, listChars):
    predictPlate = PPL.PredictPlate()
    listChars.sort(key = lambda mChar : mChar.centerX)
    flatPlateX = (listChars[0].centerX + listChars[len(listChars)-1].centerX) / 2
    flatPlateY = (listChars[0].centerY + listChars[len(listChars)-1].centerY) / 2
    coorPlateCenter = flatPlateX, flatPlateY
    PlateW = int((listChars[len(listChars) - 1].bRectX + listChars[len(listChars) - 1].bRectW - listChars[0].bRectX) * PLATE_WIDTH_PADDING_FACTOR)
    totalCharH = 0

    for i in listChars:
        totalCharH += i.bRectH

    flatAvgCharH = totalCharH / len(listChars)
    plateH = int(flatAvgCharH * PLATE_HEIGHT_PADDING_FACTOR)
    flatOpp = listChars[len(listChars) - 1].centerY - listChars[0].centerY
    flatHyp = RC.DistanceChars(listChars[0], listChars[len(listChars) - 1])
    flatCorrAngelRad = math.asin(flatOpp / flatHyp)
    flatCorrAngelDeg = flatCorrAngelRad * (180/math.pi)
    predictPlate.locPlate = (tuple(coorPlateCenter), (PlateW, plateH), flatCorrAngelDeg)
    rotMatrix = cv2.getRotationMatrix2D(tuple(coorPlateCenter), flatCorrAngelDeg, 1.0)
    h, w, numC = imgO.shape
    iRot = cv2.warpAffine(imgO, rotMatrix, (w, h))
    iCrop = cv2.getRectSubPix(iRot, (PlateW, plateH), tuple(coorPlateCenter))
    predictPlate.iPlate = iCrop

    return predictPlate


def CropPlates(img):
    ListPlates = []
    h, w, numC = img.shape

    GrayScale = np.zeros((h,w,1), np.uint8)
    thresh = np.zeros((h,w,1), np.uint8)
    contours = np.zeros((h,w,3), np.uint8)
    cv2.destroyAllWindows()

    if MainScript.showSteps == True:
        cv2.imshow('0', img)

    GrayScale, thresh = PP.preprocessing(img)

    if MainScript.showSteps == True:
        cv2.imshow('1a', GrayScale)
        cv2.imshow('1b', thresh)

    ListChars = lookupChars(thresh)

    if MainScript.showSteps == True:
        print('\n Step 2 - length of ListChars =', len(ListChars))
        contours = np.zeros((h,w,3), np.uint8)
        Conts = []

        for n in ListChars:
            Conts.append(n.contour)

        cv2.drawContours(contours, Conts, -1, MainScript.BGR_white)
        cv2.imshow('2b', contours)

    FitChars = RC.DetectFitChars(ListChars)

    if MainScript.showSteps == True:
        print('\nStep 3 - FitChars =', len(FitChars))
        contours = np.zeros((h,w,3), np.uint8)

        for i in FitChars:
            RandBlue = random.randint(0,255)
            RandGreen = random.randint(0,255)
            RandRed = random.randint(0,255)
            Conts = []

            for n in i:
                Conts.append(n.contour)

            cv2.drawContours(contours, Conts, -1, (RandBlue, RandGreen, RandRed))

        cv2.imshow('3', contours)

    for i in FitChars:
        predictPlate = gainPlate(img, i)

        if predictPlate.iPlate is not None:
            ListPlates.append(predictPlate)

    print("\n", len(ListPlates), "possibilities found!")

    if MainScript.showSteps == True:
        cv2.imshow("4a", contours)

        for i in range(len(ListPlates)):
            p2fRpt = cv2.boxPoints(ListPlates[i].locPlate)
            cv2.line(contours, tuple(p2fRpt[0]), tuple(p2fRpt[1]), MainScript.BGR_red, 2)
            cv2.line(contours, tuple(p2fRpt[1]), tuple(p2fRpt[2]), MainScript.BGR_red, 2)
            cv2.line(contours, tuple(p2fRpt[2]), tuple(p2fRpt[3]), MainScript.BGR_red, 2)
            cv2.line(contours, tuple(p2fRpt[3]), tuple(p2fRpt[0]), MainScript.BGR_red, 2)
            cv2.imshow("4a", contours)
            print("One of some predicted plate :",i,", click to continue!")
            cv2.imshow('4b', ListPlates[i].iPlate)
            cv2.waitKey(0)

        print("\nPlate detection complete!")
        cv2.waitKey(0)

    return ListPlates