import numpy as np
import cv2
import glob
import time
import sys

# Parameters
overlayAnimDir = './images/globe_png/*'
debugMode = False
showAllDebugViews = False
useRhoThetaLineMethod = False

# Initialization of globals
cap = cv2.VideoCapture(0)
imagelist = []
currentImage = 0
millis = 0
intersects = []
errorFrames = 0
frameMillis = 0
globalCounter = 0
globalTimes = []
for filename in glob.glob(overlayAnimDir):
    imagelist.append(cv2.imread(filename))


def OverlayImage (imgbg, imgfg):
    # Overlay partly transparant image over another
    nonTransPix = imgfg > 0
    imgbg[nonTransPix] = imgfg[nonTransPix]
    return imgbg

def NextOverlay (imagelist, currentImage):
    # Retrieve next overlay image from image list for animation
    overlay = imagelist[currentImage]
    currentImage = currentImage + 1
    if currentImage > len(imagelist) - 1:
        currentImage = 0
    return overlay, currentImage

def MTransform(img, M, fullImg):
    # Transform image based on transformation matrix M
    rows, cols, clrs = fullImg.shape
    imgTransform = cv2.warpPerspective(img, M, (cols, rows))
    return imgTransform

def SampleM():
    # Manually define a transformation matrix as initial state
    sx = 0.25
    sy = 0.25
    tx = 0
    ty = 0
    M = np.float32([[sx, 0, tx],[0, sy, ty],[0, 0, 1]])
    return M

def OverlayFPS(img):
    # Overlay FPS counter on image
    global millis
    color = (255, 255, 255)
    position = (0, 15)
    size = 0.5
    fps = round(1 / (time.time() * 1000 - millis) * 1000)
    millis = time.time() * 1000
    fps_text = str(int(fps)) + " FPS"
    cv2.putText(img, fps_text, position, cv2.FONT_HERSHEY_SIMPLEX, size, color)
    return img

def FilterColor(img):
    # Filter image for specific hue-saturation-value range
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Pink Post-its bounding box:
    MinHSV = np.array([203.5/255*180, 83.3, 69.6])
    MaxHSV = np.array([254.9/255*180, 221.4, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(imgHSV, MinHSV, MaxHSV)

    # AND mask (bitwise) applied to original image
    if debugMode:
        imgMaskedBGR = cv2.bitwise_and(img, img, mask = mask)
    else:
        imgMaskedBGR = None

    return imgMaskedBGR, mask

def GetEdges(img):
    # Get edges based on filtered input image
    thresLow = 100
    thresHigh = 200
    edges = cv2.Canny(img, thresLow, thresHigh)
    return edges

def FindBestLines(img):
    # Finds potential lines in the image
    thresh = 15
    rhoSize = 1
    thetaSize = np.pi / 45
    lines = cv2.HoughLines(img, rhoSize, thetaSize, thresh)
    allLines = lines

    # Pick method for line finding
    if useRhoThetaLineMethod:
        linesFiltered = FindLinesDissimilar(lines)
    else:
        linesFiltered = FindLinesIterative(lines, img, rhoSize, thetaSize, thresh)

    return (linesFiltered, allLines)

def FindLinesIterative(lines, img, rhoSize, thetaSize, thresh):
    # Iterative Hough method for finding the best lines
    linesFiltered = np.ndarray(shape=(0, 1, 2))
    imgEdit = np.copy(img)
    for i in range(0, 4):
        if lines is None:
            if debugMode:
                print("No lines found")
            return lines
        else:
            line = lines[0]
            lineExpDim = np.expand_dims(line, axis=0)
            linesFiltered = np.append(linesFiltered, lineExpDim, axis = 0)
            imgEdit = RemoveLine(imgEdit, line)
        lines = cv2.HoughLines(imgEdit, rhoSize, thetaSize, thresh)
    return linesFiltered

def RemoveLine(img, line):
    # Removes the detected line from edge image
    color = (0, 0, 0)
    width = 3
    lineExpDim = np.expand_dims(line, axis=0)
    img = DrawLinesOnImage(img, lineExpDim, width, color)
    return img

def FindLinesDissimilar(lines):
    # Similar rho-theta method for finding the best lines
    if lines == None:
        return None

    linesFiltered = np.ndarray(shape=(0, 1, 2))

    for line in lines:
        if not HasSimilarLines(line, linesFiltered):
            linesFiltered = np.append(linesFiltered, np.expand_dims(line, axis=0), axis = 0)
            if len(linesFiltered) == 4: 
                break

    return linesFiltered

def HasSimilarLines(line, fLines):
    # Checks whether there are lines in fLines with similar theta / rho
    if len(fLines) == 0:
        return False

    rhoRange = 20
    thetaRange = np.pi / 8

    rho = line[0][0]
    theta = line[0][1]
    for fLine in fLines:
        fRho = fLine[0][0]
        fTheta = fLine[0][1]
        if theta == fTheta:
            continue
        for i in range(-1, 3):
            if (fTheta - thetaRange - i * np.pi) <= theta <= (fTheta + thetaRange + i * np.pi):
                if (fRho - rhoRange) <= rho <= (fRho + rhoRange):
                    return True
    return False

def GetIntersects(lines, img, oldIntersects):
    # Given 4 lines, finds the best intersects
    intersects = np.ndarray(shape=(0, 2))
    if lines != None:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                rho1 = lines[i][0][0]
                theta1 = lines[i][0][1]
                rho2 = lines[j][0][0]
                theta2 = lines[j][0][1]
                
                intersectCandidate = GetIntersectFromPolar(rho1, theta1, rho2, theta2, img)
                if intersectCandidate != None:
                    intersects = np.vstack((intersects, intersectCandidate))

    intersects = ValidQuadrilateral(intersects)
    if intersects == None:
        if debugMode:
            print("No intersects found")
        intersects = oldIntersects
    return intersects

def GetIntersectFromPolar(rho1, theta1, rho2, theta2, img):
    # Find intersections on image of lines described by polar coordinates
    if rho1 == rho2 and theta1 == theta2:
        return None    
    a = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    if np.linalg.cond(a) < 1 / sys.float_info.epsilon:
        intersect = np.linalg.solve(a, b)
    else:
        return None
    if IsOnImage(intersect, img):
        return intersect
    return None

def IsOnImage(intersect, img):
    # Check if a point falls on (or just outside of) the image
    blur = 20
    y = img.shape[0]
    x = img.shape[1]
    if((-blur) < intersect[0] < (x + blur) and (-blur) < intersect[1] < (y + blur)):
        return True
    return False

def ValidQuadrilateral(intersects):
    # Given a set of intersects, check if we have 4 good ones, return them if so

    # Less than 4 intersects on image: return none
    if (len(intersects) < 4):
        if debugMode:
            print("Only %s intersects") % (len(intersects))
        return None

    # More than 4 intersects: use the 4 closest
    if (len(intersects) > 4):
        if debugMode:
            print("Found %s intersects, choosing closest 4") % (len(intersects))
        intersects = ClosestIntersects(intersects)

    # Sort, if sort fails (e.g. 3 intersects in a line): return none
    intersects = SortIntersects(intersects)
    if (len(intersects) < 4):
        if debugMode:
            print("Couldn't sort intersects (likely 3 on a line)")
        return None 

    # Check if opposing sides are of similar distance
    if not SimilarOpposingSides(intersects):
        if debugMode:
            print("Discarding intersects as opposing sides are dissimilar")
        return None

    return intersects

def SimilarOpposingSides(intersects):
    # Check if opposing sides are of similar distance
    tolerance = 0.12
    distLeft = IntersectDistance(intersects[0], intersects[3])
    distTop = IntersectDistance(intersects[0], intersects[1])
    distRight = IntersectDistance(intersects[1], intersects[2])
    distBottom = IntersectDistance(intersects[2], intersects[3])
    meanHeight = (distLeft + distRight) / 2
    meanWidth = (distTop + distBottom) / 2
    minWidth = meanWidth * (1 - tolerance)
    maxWidth = meanWidth * (1 + tolerance)
    minHeight = meanHeight * (1 - tolerance)
    maxHeight = meanHeight * (1 + tolerance)
    if(distLeft < minHeight or distLeft > maxHeight or distRight < minHeight or distRight > maxHeight):
        if debugMode:
            print("distLeft: %s distRight: %s minHeight: %s maxHeight: %s ") % \
                (distLeft, distRight, minHeight, maxHeight)
        return False
    if(distTop < minWidth or distTop > maxWidth or distBottom < minWidth or distBottom > maxWidth):
        if debugMode:
            print("distTop: %s distBottom: %s minWidth: %s maxWidth: %s ") % \
                (distTop, distBottom, minWidth, maxWidth)
        return False
    return True

def IntersectDistance(intersect1, intersect2):
    x1 = intersect1[0]
    x2 = intersect2[0]
    y1 = intersect1[1]
    y2 = intersect2[1]
    dist = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
    return dist

def ClosestIntersects(intersects):
    # For more than 5 intersects, get the 4 closest to each other
    dists = []
    filteredIntersects = np.ndarray(shape=(0, 2))
    xmean = np.mean(intersects[:,0])
    ymean = np.mean(intersects[:,1])
    for i in range(0, len(intersects)):
        x = intersects[i][0]
        y = intersects[i][1]
        dists.append(np.sqrt(np.square(xmean - x) + np.square(ymean - y)))
    for i in range(0, 4):
        idx = dists.index(min(dists))
        filteredIntersects = np.vstack((filteredIntersects, intersects[idx]))
        dists[idx] = float("inf")
    return filteredIntersects

def SortIntersects(its):
    # Sort intersects top-left, top-right, bottom-right, bottom-left
    sortedIntersects = np.ndarray(shape=(0, 2))

    topIts = its[its[:,0] < np.median(its[:,0])]
    bottomIts = its[its[:,0] > np.median(its[:,0])]
    topLeft = topIts[topIts[:,1] < np.mean(topIts[:,1])]
    topRight = topIts[topIts[:,1] > np.mean(topIts[:,1])]
    bottomLeft = bottomIts[bottomIts[:,1] < np.mean(bottomIts[:,1])]
    bottomRight = bottomIts[bottomIts[:,1] > np.mean(bottomIts[:,1])]
    sortedTmp = (sortedIntersects, topLeft, topRight, bottomRight, bottomLeft)   
    sortedIntersects = np.vstack(sortedTmp)
    return sortedIntersects

def DrawLinesOnImage(img, lines, width, color = (255, 255, 255)):
    # Plot lines on an image
    if lines != None:
        for rhotheta in lines:
            rho = rhotheta[0][0]
            theta = rhotheta[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(img, (x1, y1),(x2, y2), color, width)
    return img

def DrawIntersectsOnImage(img, intersects, size, width):
    # Plot the intersects on an image
    color = (255, 255, 255)
    if intersects != None:
        for intersect in intersects:
            y = int(intersect[0])
            x = int(intersect[1])
            cv2.circle(img, (y, x), size, color, width)
    return img

def ErodeDilate(img):
    # Reduces noise (small / non-uniform areas)
    size = 10
    img = cv2.erode(img, np.ones((size, size)))
    img = cv2.dilate(img, np.ones((size, size)))
    return img

def PerspectiveTransform(img, its, fullImg):
    # Transforms an image based on quadrilateral corners (its)
    if len(its) < 4:
        M = SampleM()
    else:
        imgy = img.shape[0]
        imgx = img.shape[1]
        imgRect = np.float32([[0,0],[0,imgx],[imgy,imgx],[imgy,0]])
        y1 = its[0][0]
        x1 = its[0][1]
        y2 = its[1][0]
        x2 = its[1][1]
        y3 = its[2][0]
        x3 = its[2][1]
        y4 = its[3][0]
        x4 = its[3][1]
        intersectRect = np.float32([[y1, x1], [y2, x2], [y3, x3], [y4, x4]])
        M = cv2.getPerspectiveTransform(imgRect,intersectRect)

    out = MTransform(img, M, fullImg)
    return out

def GetFrameFromWebcam():
    # Reads a frame from the webcam
    ret, frame = cap.read()
    frameResized = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    return frameResized

def StartTime():
    # Start timer
    global frameMillis
    frameMillis = time.time() * 1000 #%%%

def EndTime():
    # End timer
    global globalCounter
    currentMillis = time.time() * 1000 #%%%
    globalTimes.append(currentMillis - frameMillis)
    print(currentMillis - frameMillis)
    if globalCounter == 250:
        print globalTimes
        print np.mean(globalTimes[1:250])
        print np.std(globalTimes[1:250])
        exit()
    globalCounter = globalCounter + 1


while(True):
    
    frame = GetFrameFromWebcam()
    if debugMode:
        StartTime()
        print("Frame: %s") % (globalCounter)
    filteredFrame, mask = FilterColor(frame)
    maskClean = ErodeDilate(mask)
    edgesImg = GetEdges(maskClean)
    linesBest, linesAll = FindBestLines(edgesImg)
    intersects = GetIntersects(linesBest, frame, intersects)

    overlayRaw, currentImage = NextOverlay(imagelist, currentImage)
    overlay = PerspectiveTransform(overlayRaw, intersects, frame)
    output = OverlayImage(frame, overlay)

    if debugMode:
        debugFrame = np.copy(filteredFrame)
        debugFrame = DrawLinesOnImage(debugFrame, linesAll, 1)
        debugFrame = DrawLinesOnImage(debugFrame, linesBest, 3)
        DrawIntersectsOnImage(debugFrame, intersects, 5, 2)
        debugFrame = OverlayFPS(debugFrame)

    # Display the resulting frame
    cv2.imshow('output', output)
    
    if debugMode:
        EndTime()
        cv2.imshow('debugFrame', debugFrame)
        cv2.imwrite('debug-' + str(globalCounter) + '-img.png', output)
        cv2.imwrite('debug-' + str(globalCounter) + '-dbg.png', debugFrame)
        if showAllDebugViews:
            cv2.imshow('mask', mask)
            cv2.imshow('maskClean', maskClean)
            cv2.imshow('edgesImg', edgesImg)
        

#Release the capture
cap.release()
cv2.destroyAllWindows()