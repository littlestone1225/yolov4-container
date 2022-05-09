import os
import cv2
import numpy as np

def cropImage_PK(imgRaw, SDFactor=2.5, isLinkouRaw=True):
    rawH, rawW, rawCH = imgRaw.shape
    #print(rawW, rawH, rawCH) #test only
    defaultWidth = 640.0
    shrinkFactor = defaultWidth/rawW
    imgShrink = cv2.resize(imgRaw, (0,0), fx=shrinkFactor, fy=shrinkFactor)
    shrinkH, shrinkW, shrinkCH = imgShrink.shape
    #print(shrinkW, shrinkH, shrinkCH) #test only 
    imgGray = cv2.cvtColor(imgShrink, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("D:\\temptemp\\rrrr1.bmp",imgGray)#test only
    grayH, grayW = imgGray.shape #since it become dim =2, the thir dimension cannot get
    #print(grayW, grayH) #test only
    thrOtsu, imgOtsu = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #print (thrOtsu)#test only
    #cv2.imwrite("D:\\temptemp\\rrrr2.bmp",imgOtsu)#test only

    if True == isLinkouRaw:
        imgHSV=cv2.cvtColor(imgShrink, cv2.COLOR_BGR2HSV) 
        yellowTapeImg = np.zeros((grayH, grayW, 1), dtype = "uint8")
        for row in range(shrinkH):
            for col in range(shrinkW):
                value = imgGray[row, col]
                hsv = imgHSV[row, col]
                hue = 2*hsv[0] #yellow colour at hue =60
                sat = hsv[1]
                if value>thrOtsu and hue>=40 and hue<=80 and sat>=80:
                #satisfy Otsu thresholding, yellowish colour and enough saturation; those values are set imperically
                    yellowTapeImg[row, col]=255
                    
        #cv2.imwrite("D:\\temptemp\\rrrr2-1.bmp",yellowTapeImg)#test only
        yellowTapeImg2 = cv2.blur(yellowTapeImg, (5,5))
        #cv2.imwrite("D:\\temptemp\\rrrr2-2.bmp", yellowTapeImg2)#test only
        ret, yellowTapeImg3 = cv2.threshold(yellowTapeImg2, 128, 255, cv2.THRESH_BINARY)
        #cv2.imwrite("D:\\temptemp\\rrrr2-3.bmp", yellowTapeImg3)#test only
        
        #find contour!!!
        (_, contours,_) = cv2.findContours(yellowTapeImg3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #print("contours number:",len(contours))#test only

        thirdStripRatio = (53.0 * 15.0) / (640.0 * 526.0)#roughly 1/3 strip size 53x15 pixels; (53*15)/(640*526)
        AREA_THR = thirdStripRatio*shrinkW*shrinkH 

        newContours = []#set an empty  array
        for contourCnt in range(len(contours)):
            area = abs(cv2.contourArea(contours[contourCnt]))
            if area>=AREA_THR:
                newContours.append(contours[contourCnt])
        #print("new contours number:", len(newContours))#test only
        #print("AREA_THR:", AREA_THR)#test only

        negateYellowTapeImg = np.full((grayH, grayW, 1), 255, dtype = "uint8") #fill 255 for all elements
        cv2.drawContours(negateYellowTapeImg, newContours, -1,(0),-1)
        #cv2.imwrite("d:\\temptemp\\rrrr2-4.bmp", negateYellowTapeImg)#test only
        cmpMat = cv2.bitwise_and(imgOtsu, negateYellowTapeImg)
        #cv2.imwrite("d:\\temptemp\\rrrr2-5.bmp", cmpMat)#test only
        imgOtsu = cmpMat.copy()#copy to imgOtsu
        
    locations = cv2.findNonZero(imgOtsu)
    dim0, dim1, dim2 = locations.shape

    if dim0<100:# samples are too few, cause early return
        left = 0
        top= 0
        right = rawW-1
        bottom = rawH-1
        return left, top, right, bottom
   
    #print("dim0 dim1 dim2:", dim0,dim1,dim2) #test only
    newLocations = np.reshape(locations, (dim0,2,1), 'C')
    new_a, new_b, new_c = newLocations.shape
    #print("new_a, new_b, new_c:", new_a,new_b,new_c) #test only
    points64F = newLocations.astype('float64')
    mean, evectors = cv2.PCACompute(points64F, mean=None)
    mean2 = np.mean(points64F, axis=0).reshape(1,-1)#same as mean
    #print("means2:", mean2)# test only

    ###########################################################################
    #compute eigenValue and eigenVector
    covar, mean =cv2.calcCovarMatrix(points64F, mean, cv2.COVAR_SCALE|cv2.COVAR_ROWS|cv2.COVAR_NORMAL)
    retval, eVal, eVec = cv2.eigen(covar)
    #print("eigenValue", eVal)#test only
    #print("eigneVector", eVec)#test only
    eVal_sqrt = cv2.sqrt(eVal)
    ###########################################################################

    #SDFactor = 2.5 #take 2.5 SD as default
    extraFactor =1.1#where the eVal_sqrt[0]/eVal_sqrt[1]>3 (too skew), multiply extraFactor
    
    pc_angle = 180*np.arctan2(eVec[0][1], eVec[0][0])/np.pi

    if pc_angle > 0:
        if pc_angle>45.0 and pc_angle<=135:
            isVerticalOrientated = True
        else:
            isVerticalOrientated = False
    else:
        if pc_angle<-45.0 and pc_angle>=-135:
            isVerticalOrientated = True
        else:
            isVerticalOrientated = False

    centre = (mean[0][0], mean[0][1])#get fg centre
    cropHalfWidth = 0;
    cropHalfHeight = 0;

    #get correct halfWidth and halfHeight depends on isVerticalOrientated  
    if isVerticalOrientated:
        if eVal_sqrt[0]/eVal_sqrt[1]>3.0:#very skew
            cropHalfWidth = int(0.5 + SDFactor*extraFactor*eVal_sqrt[1])
            cropHalfHeight = int(0.5 + SDFactor*eVal_sqrt[0])
        else:
            cropHalfWidth = int(0.5 + SDFactor*eVal_sqrt[1])
            cropHalfHeight = int(0.5 + SDFactor*eVal_sqrt[0])
    else:
        if eVal_sqrt[0]/eVal_sqrt[1]>3.0:#very skew
            cropHalfWidth = int(0.5 + SDFactor*eVal_sqrt[0]);
            cropHalfHeight = int(0.5 + SDFactor*extraFactor*eVal_sqrt[1]);
        else:
            cropHalfWidth = int(0.5 + SDFactor*eVal_sqrt[0]);
            cropHalfHeight = int(0.5 + SDFactor*eVal_sqrt[1]);
            
    #get correct crop boundary
    left = centre[0]-cropHalfWidth
    right = centre[0]+cropHalfWidth
    top = centre[1]-cropHalfHeight
    bottom = centre[1]+cropHalfHeight

    #out of image boundary handling
    if left<0:
        left = 0
    if top<0:
        top = 0
    if right>shrinkW-1:
        right = shrinkW-1
    if bottom>shrinkH-1:
        bottom = shrinkH-1

    #blow up the size to fit the origina image size
    left /= shrinkFactor
    top /= shrinkFactor
    right /= shrinkFactor
    bottom /= shrinkFactor

    #become integer
    left = int(0.5+left)
    top = int(0.5+top)
    right = int(0.5+right)
    bottom = int(0.5+bottom)

    #draw to show to validate test only
    #imgDraw=imgRaw.copy()#deep copy
    #cv2.rectangle(imgDraw, (left, top), (right, bottom), (0,255,255), 5 )
    #cv2.rectangle(imgDraw, (0,0), (rawW-1, rawH-1), (255,0,255), 20 )
    #cv2.imwrite("D:\\temptemp\\rrrr5.bmp", imgDraw)#test only
    #print("complete!")#test only

    return left, top, right, bottom


        



