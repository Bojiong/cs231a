import numpy as np
import cv2
import glob
import time
import sys

cap = cv2.VideoCapture(0)
overlayAnimDir = './images/globe_png/*'
imagelist = []
currentImage = 0
millis = 0
intersects = []
for filename in glob.glob(overlayAnimDir):
    imagelist.append(cv2.imread(filename))


def OverlayImage (imgbg, imgfg):
    # Overlay partly transparant image over another
    imgbg[imgfg > 0] = imgfg[imgfg > 0]
    return imgbg

def Overlay3DPoints (img):
    # Overlay 3D points (not used)
    Pts2D = np.ones((6, 3))
    Pts2D[:,1] = 20
    Pts2D[:,2] = range(0, 6, 10)
    r = np.identity(3) # rotation vector
    t = np.array([0,0,0], np.float) # translation vector
    fx = fy = 1.0
    cx = cy = 0.0
    cameraMatrix = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
    Pts3D = cv2.projectPoints(Pts2D, r, v, cameraMatrix, None)
    for n in Pts3D:
        #Pt3Dx = int(Pts3D[0][n][0][0])
        #Pt3Dy = int(Pts3D[0][n][0][1])
        cv2.circle(img, (Pt3Dx, Pt3Dy), 3, (200, 200, 200))
    return img

def NextOverlay (imagelist, currentImage):
    # Retrieve next overlay image from image list for animation
    overlay = imagelist[currentImage]
    currentImage = currentImage + 1
    if currentImage > len(imagelist) - 1:
        currentImage = 0
    return overlay, currentImage

def MTransform(img, M, cols, rows):
    # Transform image based on transformation matrix M
    imgTransform = cv2.warpPerspective(img,M,(cols,rows))
    return imgTransform

def SampleM():
    # Manually define a transformation matrix
    sx = 0.25
    sy = 0.25
    tx = 0
    ty = 0
    M = np.float32([[sx,0,tx],[0,sy,ty],[0,0,1]])
    #M = np.float32([[sx,0.1,tx],[0.5,sy,ty],[0.00001,-0.0004,1]])
    return M

def OverlayFPS(img):
    # Overlay FPS counter on image
    global millis
    fps = round(1 / (time.time() * 1000 - millis) * 1000)
    millis = time.time() * 1000
    fps_text = str(int(fps)) + " FPS"
    cv2.putText(img, fps_text, (0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    return img

def FilterColor(img):
    # Filter image for specific hue-saturation-value range
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Conservative:
    #MinHSV = np.array([207.9/255*180, 106.4, 91.2])
    #MaxHSV = np.array([250.4/255*180, 206.4, 255])
    # Wider:
    #MinHSV = np.array([207.9/255*180, 97.2, 91.2])
    #MaxHSV = np.array([250.4/255*180, 207.6, 255])
    # Widest:
    MinHSV = np.array([203.5/255*180, 83.3, 69.6])
    MaxHSV = np.array([254.9/255*180, 221.4, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(imgHSV, MinHSV, MaxHSV)

    # AND mask (bitwise) applied to original image
    imgMaskedBGR = cv2.bitwise_and(img, img, mask = mask)
    return imgMaskedBGR, mask

def GetEdges(img):
    # Get edges based on filtered input image
    edges = cv2.Canny(img,100,200)
    return edges

def FindBestLines(img):
    # Finds potential lines in the image
    # Filters to 4 lines best meeting several basic constraints
    thresh = 15
    lines = cv2.HoughLines(img, 1, np.pi/45, thresh)
    if lines is None:
        linesFiltered = lines
    else:
        linesFiltered = FilterBestLines(lines)

    return (linesFiltered, lines)


def FilterBestLines(lines):
    # Given a number of lines, finds the most promising 4
    linesFiltered = np.ndarray(shape=(0, 1, 2))

    for line in lines:
        if not HasSimilarLines(line, linesFiltered):
            linesFiltered = np.append(linesFiltered, np.expand_dims(line, axis=0), axis = 0)
            if len(linesFiltered) == 4: 
                break

    return linesFiltered

def HasSimilarLines(line, fLines):
    # Checks whether there are lines in fLines very similar to line
    if len(fLines) == 0:
        return False

    rhoRange = 20
    thetaRange = np.pi / 4

    rho = line[0][0]
    theta = line[0][1]
    for fLine in fLines:
        fRho = fLine[0][0]
        fTheta = fLine[0][1]
        #print "Theta: %s fTheta: %s Rho: %s fRho: %s" % (theta, fTheta, rho, fRho)
        if (fRho - rhoRange) <= rho <= (fRho + rhoRange):
            for i in range(-1, 3):
                if (fTheta - thetaRange + i * np.pi) <= theta <= (fTheta +thetaRange + i * np.pi):
                    return True
    return False

def GetIntersects(lines, img):
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
    return intersects

def GetIntersectFromPolar(rho1, theta1, rho2, theta2, img):
    # Find intersections of lines described by polar coordinates
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
    blur = 5
    y = img.shape[0]
    x = img.shape[1]
    if(intersect[0] > (-blur) and intersect[0] < (x + blur) and intersect[1] > (-blur) and intersect[1] < (y + blur)):
        return True
    return False

def ValidQuadrilateral(intersects):
    # Given a set of intersects, check if we have 4 good ones, return them if so
    minDist = 10
    filteredIntersects = np.ndarray(shape=(0, 2))

    if (len(intersects) < 4):
        return None

    # Filter out if one point is too far out
    xmean = np.mean(intersects[:,0])
    ymean = np.mean(intersects[:,1])
    xstd = np.std(intersects[:,0])
    ystd = np.std(intersects[:,1])
    for intersect in intersects:
        x = intersect[0]
        y = intersect[1]
        if (x >= (xmean - 2*xstd) and x <= (xmean + 2*xstd) and y >= (ymean - 2*ystd) and y <= (ymean + 2*ystd)):
            filteredIntersects = np.vstack((filteredIntersects, intersect))

    # Filter out if two points are too near
    intersectCandidates = np.copy(filteredIntersects)
    filteredIntersects = np.ndarray(shape=(0, 2))
    for i in range(len(intersectCandidates)):
        badCandidate = False
        for j in range(i + 1, len(intersectCandidates)):  
            i1x = intersectCandidates[i][0]
            i1y = intersectCandidates[i][1]
            i2x = intersectCandidates[j][0]
            i2y = intersectCandidates[j][1]
            if((i2x - minDist) <= i1x <= (i2x + minDist) and (i2y - minDist) <= i1y <= (i2y + minDist)):
                badCandidate = True
        if(badCandidate == False):
            filteredIntersects = np.vstack((filteredIntersects, intersectCandidates[i]))

    if (len(filteredIntersects) != 4):
        return None

    filteredIntersects = SortIntersects(filteredIntersects)

    return filteredIntersects

def SortIntersects(its):
    # Sort intersects top-let, top-right, bottom-right, bottom-left
    sortedIntersects = np.ndarray(shape=(0, 2))

    topIts = its[its[:,0] < np.median(its[:,0])]
    bottomIts = its[its[:,0] > np.median(its[:,0])]
    topLeft = topIts[topIts[:,1] < np.mean(topIts[:,1])]
    topRight = topIts[topIts[:,1] > np.mean(topIts[:,1])]
    bottomLeft = bottomIts[bottomIts[:,1] < np.mean(bottomIts[:,1])]
    bottomRight = bottomIts[bottomIts[:,1] > np.mean(bottomIts[:,1])]
    
    sortedIntersects = np.vstack((sortedIntersects, topLeft, topRight, bottomRight, bottomLeft))
    
    return sortedIntersects


def LinesOnImage(img, lines, width):
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
            cv2.line(img, (x1, y1),(x2, y2),(255, 255, 255), width)

    return img

def IntersectsOnImage(img, intersects, size, width):
    # Plot the intersects on an image
    if intersects != None:
        for intersect in intersects:
            cv2.circle(img, (int(intersect[0]), int(intersect[1])), size, (255, 255, 255), width)
    return img

def DilateErode(img):
    # Reduces noise (not used)
    size = 5
    img = cv2.erode(img, np.ones((size, size)))
    img = cv2.dilate(img, np.ones((size, size)))
    return img

def PerspectiveTransform(img, its, fullimg):
    # Transforms an image based on quadrialateral corners (its)
    if len(its) < 4:
        M = SampleM()
    else:
        imgy = img.shape[0]
        imgx = img.shape[1]
        imgRect = np.float32([[0,0],[0,imgx],[imgy,imgx],[imgy,0]])
        intersectRect = np.float32([[its[0][0], its[0][1]], [its[1][0], its[1][1]], [its[2][0], its[2][1]], [its[3][0], its[3][1]]])
        M = cv2.getPerspectiveTransform(imgRect,intersectRect)
    
    rows,cols,clrs = fullimg.shape
    out = MTransform(img, M, cols, rows)
    return out


while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    rows, cols, clrs = frame.shape
    filteredFrame, mask = FilterColor(frame)
    #strongMask = DilateErode(mask)
    edgesImg = GetEdges(mask)
    linesBest, linesAll = FindBestLines(edgesImg)
    filteredFrame = LinesOnImage(filteredFrame, linesAll, 1)
    filteredFrame = LinesOnImage(filteredFrame, linesBest, 3)
    newIntersects = GetIntersects(linesBest, frame)
    if newIntersects != None:
        intersects = newIntersects
    filteredFrame = IntersectsOnImage(filteredFrame, intersects, 10, 2)
    #frame = Overlay3DPoints(frame)

    overlayRaw, currentImage = NextOverlay(imagelist, currentImage)
    overlay = PerspectiveTransform(overlayRaw, intersects, frame)
    
    frameWithOverlay = OverlayImage(frame, overlay)
    frameWithOverlay = OverlayFPS(frameWithOverlay)

    # Display the resulting frame
    cv2.imshow('filteredFrame', filteredFrame)
    cv2.imshow('output', frameWithOverlay)
    #cv2.imshow('mask', mask)
    #cv2.imshow('strongmask', strongMask)
    #cv2.imshow('lines',linesimg)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()