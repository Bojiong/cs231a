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
errorFrames = 0
for filename in glob.glob(overlayAnimDir):
    imagelist.append(cv2.imread(filename))


def OverlayImage (imgbg, imgfg):
    # Overlay partly transparant image over another
    imgbg[imgfg > 0] = imgfg[imgfg > 0]
    return imgbg

def Overlay3DPoints (img):
    # Overlay 3D points
    # UNUSED
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
    linesFiltered = np.ndarray(shape=(0, 1, 2))
    cv2.imwrite('edgeFrame-start.png', img)
    for i in range(0, 4):
        lines = cv2.HoughLines(img, 1, np.pi/45, thresh)
        if lines is None:
            return (lines, lines)
        else:
            linesFiltered = np.append(linesFiltered, np.expand_dims(lines[0], axis=0), axis = 0)
            img = DrawLinesOnImage(img, np.expand_dims(lines[0], axis=0), 5, (0, 0, 0))
            cv2.imwrite('edgeFrame-' + str(i) + '.png', img)

    exit()
    return (linesFiltered, lines)


def FilterBestLines(lines):
    # Given a number of lines, finds the most promising 4
    # UNUSED
    linesFiltered = np.ndarray(shape=(0, 1, 2))

    for line in lines:
        if not HasSimilarLines(line, linesFiltered):
            linesFiltered = np.append(linesFiltered, np.expand_dims(line, axis=0), axis = 0)
            if len(linesFiltered) == 4: 
                break

    return linesFiltered

def HasSimilarLines(line, fLines):
    # Checks whether there are lines in fLines at similar distance but slightly different angle
    # UNUSED
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
        #print "Theta: %s fTheta: %s Rho: %s fRho: %s" % (theta, fTheta, rho, fRho)
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
    blur = 10
    y = img.shape[0]
    x = img.shape[1]
    if(intersect[0] > (-blur) and intersect[0] < (x + blur) and intersect[1] > (-blur) and intersect[1] < (y + blur)):
        return True
    return False

def ValidQuadrilateral(intersects):
    # Given a set of intersects, check if we have 4 good ones, return them if so

    # Less than 4 intersects on image: return none
    if (len(intersects) < 4):
        #print("Less than 4")
        return None

    # More than 4 intersects: use the 4 closest
    if (len(intersects) > 4):
        intersects = ClosestIntersects(intersects)

    # Sort, if sort fails (e.g. 3 intersects in a line): return none
    intersects = SortIntersects(intersects)
    if (len(intersects) < 4):
        #print("Unsortable")
        return None 

    # Check if opposing sides are of similar distance
    if not SimilarOpposingSides(intersects):
        #print("Dissimilar")
        return None

    return intersects

def SimilarOpposingSides(intersects):
    # Check if opposing sides are of similar distance
    tolerance = 0.1
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
        #print("DistLeft: %s DistRight: %s minHeight: %s maxHeight: %s ") % (distLeft, distRight, minHeight, maxHeight)
        return False
    if(distTop < minWidth or distTop > maxWidth or distBottom < minWidth or distBottom > maxWidth):
        #print("DistTop: %s DistBottom: %s minWidth: %s maxWidth: %s ") % (distTop, distBottom, minWidth, maxWidth)
        return False
    
    return True

def IntersectDistance(intersect1, intersect2):
    dist = np.sqrt(np.square(intersect1[0] - intersect2[0]) + np.square(intersect1[1] - intersect2[1]))
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
    
    sortedIntersects = np.vstack((sortedIntersects, topLeft, topRight, bottomRight, bottomLeft))
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
    if intersects != None:
        for intersect in intersects:
            cv2.circle(img, (int(intersect[0]), int(intersect[1])), size, (255, 255, 255), width)
    return img

def DilateErode(img):
    # Reduces noise
    # UNUSED
    size = 5
    img = cv2.erode(img, np.ones((size, size)))
    img = cv2.dilate(img, np.ones((size, size)))
    return img

def PerspectiveTransform(img, its, fullimg):
    # Transforms an image based on quadrilateral corners (its)
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

def GetFrameFromVideo():
    ret, frame = cap.read()
    frameResized = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    # rows, cols, clrs = frame.shape
    return frameResized

while(True):
    
    frame = GetFrameFromVideo()
    filteredFrame, mask = FilterColor(frame)
    mask = DilateErode(mask)
    edgesImg = GetEdges(mask)
    linesBest, linesAll = FindBestLines(edgesImg)
    intersects = GetIntersects(linesBest, frame, intersects)

    #filteredFrame = DrawIntersectsOnImage(filteredFrame, intersects, 10, 2)
    overlayRaw, currentImage = NextOverlay(imagelist, currentImage)
    overlay = PerspectiveTransform(overlayRaw, intersects, frame)
    output = OverlayImage(frame, overlay)
    #output = OverlayFPS(output)
    
    debugFrame = np.copy(filteredFrame)
    debugFrame = DrawLinesOnImage(debugFrame, linesAll, 1)
    debugFrame = DrawLinesOnImage(debugFrame, linesBest, 3)
    #debugFrame = OverlayFPS(debugFrame)

    # Display the resulting frame
    #cv2.imshow('filteredFrame', filteredFrame)
    #cv2.imshow('edgesImg', edgesImg)
    cv2.imshow('output', output)
    cv2.imshow('debugFrame', debugFrame)
    #cv2.imshow('mask', mask)
    #cv2.imshow('strongmask', strongMask)
    #cv2.imshow('lines',linesimg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()