import cv2
import numpy as np
import math
import random
import os

import MainScript
import EngineFiles.PredictChar as PC
import EngineFiles.Preprocessing as PP

MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100

knn_model = cv2.ml.KNearest_create()

def KNNmodeling():
        collectiveContours = []
        validContours = []

        try:
                arrClassifications = np.loadtxt('TrainArrFiles/classifications.txt', np.float32)
                arrFlattndImgs = np.loadtxt('TrainArrFiles/flattened_images.txt', np.float32)
        except:
                print('Unable to open classifications and/or flattened images model files!')
                os.system('pause')
                return False

        arrClassifications = arrClassifications.reshape((arrClassifications.size,1))
        knn_model.setDefaultK(1)
        knn_model.train(arrFlattndImgs, cv2.ml.ROW_SAMPLE, arrClassifications)

        return True


def checkChar(Char):
        if (Char.bRectArea > MIN_PIXEL_AREA and\
        Char.bRectW > MIN_PIXEL_WIDTH and\
        Char.bRectH > MIN_PIXEL_HEIGHT and\
        Char.FlatAspRat > MIN_ASPECT_RATIO and\
        Char.FlatAspRat < MAX_ASPECT_RATIO):
                return True
        else:
                return False


def DistanceChars(first, second):
        x = abs(first.centerX - second.centerX)
        y = abs(first.centerY - second.centerY)
        return math.sqrt((x**2) + (y**2))


def AngleChars(first, second):
        FlatAdj = float(abs(first.centerX - second.centerX))
        FlatOpp = float(abs(first.centerY - second.centerY))

        if FlatAdj != 0.0:
                FlatAngleRad = math.atan(FlatOpp/FlatAdj)
        else:
                FlatAngleRad = 1.5708

        return FlatAngleRad * (180.0/math.pi)


def DetectCharinChars(Char, listChars):
        CharInChars = []

        for i in listChars:

                if i == Char:
                        continue

                FlatDistance = DistanceChars(Char, i)
                FlatAngle = AngleChars(Char, i)
                FlatChangeArea = float(abs(i.bRectArea - Char.bRectArea)) / float(Char.bRectArea)
                FlatChangeW = float(abs(i.bRectW - Char.bRectW)) / float(Char.bRectW)
                FlatChangeH = float(abs(i.bRectH - Char.bRectH)) / float(Char.bRectH)

                if (FlatDistance < (Char.FlatDiagS * MAX_DIAG_SIZE_MULTIPLE_AWAY) and\
                        FlatAngle < MAX_ANGLE_BETWEEN_CHARS and\
                        FlatChangeArea < MAX_CHANGE_IN_AREA and\
                        FlatChangeW < MAX_CHANGE_IN_WIDTH and\
                        FlatChangeH < MAX_CHANGE_IN_HEIGHT):
                        CharInChars.append(i)

        return CharInChars


def DetectFitChars(listChars):
        ListOfListsFitChars = []

        for a in listChars:
                ListOfFitChars = DetectCharinChars(a, listChars)
                ListOfFitChars.append(a)

                if len(ListOfFitChars) < MIN_NUMBER_OF_MATCHING_CHARS:
                        continue

                ListOfListsFitChars.append(ListOfFitChars)
                CurrentFitDeleted = []
                CurrentFitDeleted = list(set(listChars) - set(ListOfFitChars))

                recursiveListOfListsFitChars = DetectFitChars(CurrentFitDeleted)

                for n in recursiveListOfListsFitChars:
                        ListOfListsFitChars.append(n)

                break

        return ListOfListsFitChars


def checkChar(cChar):
        if (cChar.bRectArea > MIN_PIXEL_AREA and\
                cChar.bRectW > MIN_PIXEL_WIDTH and\
                cChar.bRectH > MIN_PIXEL_HEIGHT and\
                cChar.FlatAspRat > MIN_ASPECT_RATIO and\
                cChar.FlatAspRat < MAX_ASPECT_RATIO):
                return True
        else:
                return False


def scrapCharsPlate(grays, thresh):
        lChars, contours, threshC = [], [], thresh.copy()
        contours, npH = cv2.findContours(threshC, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
                pc = PC.PredictChar(c)

                if checkChar(pc):
                        lChars.append(pc)

        return lChars


def delOverLapChars(lChars):
        lMcDel = list(lChars)

        for current in lChars:
                for other in lChars:
                        if current != other:
                                if DistanceChars(current, other) < current.FlatDiagS * MIN_DIAG_SIZE_MULTIPLE_AWAY:
                                        if current.bRectArea < other.bRectArea:
                                                if current in lMcDel:
                                                        lMcDel.remove(current)
                                        else:
                                                if other in lMcDel:
                                                        lMcDel.remove(other)
        return lMcDel


def CharsPrediction(thresh, lChars):
        predictChars = ''
        h, w = thresh.shape
        iThreshC = np.zeros((h, w, 3), np.uint8)
        lChars.sort(key = lambda z : z.centerX)
        cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR, iThreshC)

        for i in lChars:
                node1 = (i.bRectX, i.bRectY)
                node2 = ((i.bRectX + i.bRectW), (i.bRectY + i.bRectH))
                cv2.rectangle(iThreshC, node1, node2, MainScript.BGR_green, 2)
                iROI = thresh[i.bRectY : i.bRectY + i.bRectH, i.bRectX : i.bRectX + i.bRectW]
                iROIresize = cv2.resize(iROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))
                iROIreshape = np.float32(iROIresize.reshape(1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))

                retval, results, neigh, dists = knn_model.findNearest(iROIreshape, k=1)
                pChar = str(chr(int(results[0][0])))
                predictChars += pChar

        if MainScript.showSteps == True:
                cv2.imshow('10', iThreshC)

        return predictChars


def GetCharsInPlate(listPlates):
        plateC, iContours, contours = 0, None, []

        if len(listPlates) == 0:
                return listPlates

        for i in listPlates:
                i.iGrayScale, i.iThresh = PP.preprocessing(i.iPlate)

                if MainScript.showSteps == True:
                        cv2.imshow('5a',i.iPlate)
                        cv2.imshow('5b',i.iGrayScale)
                        cv2.imshow('5c',i.iThresh)

                i.iThresh = cv2.resize(i.iThresh, (0,0), fx=1.6, fy=1.6)
                threshVal, i.iThresh = cv2.threshold(i.iThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                if MainScript.showSteps == True:
                        cv2.imshow('5d', i.iThresh)

                charsInPlate = scrapCharsPlate(i.iGrayScale, i.iThresh)

                if MainScript.showSteps == True:
                        h, w, numC = i.iPlate.shape
                        iContours = np.zeros((h, w, 3), np.uint8)
                        del contours[:]

                        for n in charsInPlate:
                                contours.append(n.contour)

                        cv2.drawContours(iContours, contours, -1, MainScript.BGR_white)
                        cv2.imshow('6', iContours)

                FitChars = DetectFitChars(charsInPlate)

                if MainScript.showSteps == True:
                        iContours = np.zeros((h, w, 3), np.uint8)
                        del contours[:]

                        for n in FitChars:
                                randBlue = random.randint(0,255)
                                randGreen = random.randint(0,255)
                                randRed = random.randint(0,255)

                                for j in n:
                                        contours.append(j.contour)

                                cv2.drawContours(iContours, contours, -1, (randBlue, randGreen, randRed))

                        cv2.imshow('7', iContours)

                if len(FitChars) == 0:
                        if MainScript.showSteps == True:
                                print("Chars found in plate number ",plateC," = (none)")
                                plateC += 1
                                cv2.destroyWindow("8")
                                cv2.destroyWindow("9")
                                cv2.destroyWindow("10")
                                cv2.waitKey(0)

                        i.sChars = ""
                        continue

                for k in range(len(FitChars)):
                        FitChars[k].sort(key = lambda z : z.centerX)
                        FitChars[k] = delOverLapChars(FitChars[k])

                if MainScript.showSteps == True:
                        iContours = np.zeros((h, w, 3), np.uint8)

                        for n in FitChars:
                                randBlue = random.randint(0,255)
                                randGreen = random.randint(0,255)
                                randRed = random.randint(0,255)
                                del contours[:]

                                for j in n:
                                        contours.append(j.contour)

                                cv2.drawContours(iContours, contours, -1, (randBlue, randGreen, randRed))

                        cv2.imshow('8', iContours)

                LLLChars = 0
                ILLChars = 0

                for n in range(len(FitChars)):
                        if len(FitChars[n]) > LLLChars:
                                LLLChars = len(FitChars[n])
                                ILLChars = n

                possibleFitChars = FitChars[ILLChars]

                if MainScript.showSteps == True:
                        iContours = np.zeros((h, w, 3), np.uint8)
                        del contours[:]

                        for j in possibleFitChars:
                                contours.append(j.contour)

                        cv2.drawContours(iContours, contours, -1, MainScript.BGR_white)
                        cv2.imshow('9', iContours)

                i.sChars = CharsPrediction(i.iThresh, possibleFitChars)

                if MainScript.showSteps == True:
                        print("Predicted characters on plate ", plateC," =", i.sChars, ". Press any key to contrinue...")
                        plateC += 1
                        cv2.waitKey(0)

        if MainScript.showSteps == True:
                print("Plat number extraction complete!")
                cv2.waitKey(0)

        return listPlates