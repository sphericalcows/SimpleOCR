# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 19:26:56 2014


@author: Freeman
"""
#-------------- Dependencies ----------------
import numpy as np
import cv2
import glob
import itertools
import sys
import pickle

global epsilon
epsilon=10

#--------------  Functions   ----------------
# angle_cos
# function used to test angles to determine if a found contour is retangular
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

# filterImage
# function used to filter image files for better rectangle and character
#       recognition. Optional parameters:
#         thresh= integer: threshold for filtering, 
#         dilate= integer: pixels to dilate the black part of the image
def filterImage(inputImage,thresh=140,dilate=0):
    img = cv2.GaussianBlur(inputImage,(3,3),100)
    ret,th1 = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)
    kernel = np.ones((4,4),np.uint8)
    closeImg = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
    openImg = cv2.morphologyEx(closeImg, cv2.MORPH_OPEN, kernel)
    if dilate!=0:
        openImg = cv2.erode(openImg,kernel,iterations = dilate)
    return openImg
    
# pseudoMetricCompare
#pseudoComparitor compares two contour values with a pseudometric order
#    eps is the distance within which coordinates are equal
def pseudoMetricCompare(a,b):
    global epsilon
    y0 = a[0][1]
    y1 = b[0][1]
    x0 = a[0][0]
    x1 = b[0][0]
    if (y0 - y1) > epsilon:
        return 1
    if (y1 - y0) > epsilon:
        return -1
    if (x0 - x1) > epsilon:
        return 1
    if (x1 - x0) > epsilon:
        return -1
    return 0

"""
FindRectangles (inputImage,outputList)
optional parameters:    minAreaSize =  fraction of area of page that a retangle
                            must occupy to be counted
                        thresh = threshold for converting to black and white
Function to find rectangle guides in image file.
Called to create template
Also used to find guide squares for extracting images
"""
def FindRectangles(inputImage,outputList,minAreaSize=0.0008,thresh=140):
    orig = inputImage
    h,w = orig.shape[:2]
    approxDPI = h/11
    minArea = int((approxDPI*approxDPI)*minAreaSize)
    global epsilon
    epsilon = approxDPI/25

    openImg = filterImage(orig)

    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(openImg,mask,(10,10),0,250,255)

    contours, hierarchy = cv2.findContours(openImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
#   This code segment taken from openCV example program squares.py    
    squares = []
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > minArea and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
            if max_cos < 0.1:
                squares.append(cnt)        
#sort the squares array to give top -> down, left -> right
#  the pseudoMetricCompare is needed because we want to consider two elements as equal if
#  they are within some distance, eps, of each other (here eps=length of page/110 or about 0.1")
    squares.sort(pseudoMetricCompare)

#load into output array
    #Extract the image for the ith character
    for i in xrange(0, len(squares)): #later change to len(squares)-1):
        cnt = squares[i]
        #Extract the image for the ith character
        maxX = max(zip(*cnt)[0])
        maxY = max(zip(*cnt)[1]) 
        minX = min(zip(*cnt)[0])
        minY = min(zip(*cnt)[1])
        outputList.append([[minX,minY],[maxX,maxY]])

    return squares

#------------------------------------------
def makePixelImage(inputImg):
    h = inputImg.shape[0]
    w = inputImg.shape[1]
    chopx = w / 3.0
    chopy = h / 5.0
    pixelImg = cv2.resize(inputImg,(3,5),0,0,cv2.INTER_CUBIC)
    for r in range(0,5):
        rmin=int(r*chopy)
        rmax=int((r+1)*chopy)
        for c in range(0,3):
            cmin = int(c*chopx)
            cmax = int((c+1)*chopx)
            pixelBlock = inputImg[rmin:rmax,cmin:cmax]
            flatBlock = list(itertools.chain.from_iterable(pixelBlock))
            avePixel = sum(flatBlock)/len(flatBlock)
            #consider applying a scaling to the pixel
            pixelImg[r,c] = int(avePixel)
    return pixelImg


if __name__ == '__main__':

#Open Template File and Extract Squares    
    templateImg = cv2.imread('C:\Users\Freeman\Documents\OMR\NumericTrainingV3_template.png',0)
    template = []
    squares = FindRectangles(templateImg,template)
    if len(template) == 0:
        print "error in reading template"
        sys.exit(1)

    digitLst=[]
    targetLst=[]
    chrtitle = ""
    
    for filename in glob.glob("C:\Users\Freeman\Documents\OMR\FurtherScans-*.png"):
# training data from "C:\Users\Freeman\Documents\OMR\NumericTrainingV3-*.png"        
# testing data from "C:\Users\Freeman\Documents\OMR\FurtherScans-*.png"
        print filename
        formOrig = cv2.imread(filename,0)
        # Filter image
        formImg = filterImage(formOrig)
        
        #find rectangles (first and last for offset adjustment - currently using only 1st)
        formRects =[]
        formSquares = FindRectangles(formImg, formRects)
        
        shiftX = formRects[0][0][0] - template[0][0][0]
        shiftY = formRects[0][0][1] - template[0][0][1]
        
        # Filter image again for characters
        formImg = filterImage(formOrig, thresh=130, dilate=2)
        
        #process digits out of filtered image file based on template
                
        for i in xrange(1, len(template)-1): #later change to len(template)-1):
            rect = template[i]
            #Extract the image for the ith character
            minX = rect[0][0]+ shiftX - 2
            minY = rect[0][1]+ shiftY - 2
            maxX = rect[1][0]+ shiftX + 2
            maxY = rect[1][1]+ shiftY + 2

            digitImg = formImg[minY:maxY,minX:maxX]
            pixImg = makePixelImage(digitImg)
            digitLst.append(pixImg)
            targetLst.append((i-1)%21)
    #end loops, now put into one data structure and save it                    
    dataSet = zip(targetLst,np.invert(digitLst))

    outfile = open("C:\Users\Freeman\Documents\OMR\DigitData.txt", "w")    
    pickle.dump(dataSet,outfile)
    outfile.close()
    print "Processed ",len(dataSet), " character records"