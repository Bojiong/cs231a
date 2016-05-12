import cv2
import time
import math
import numpy as np
import sift

capture = cv2.VideoCapture(0)
#print capture.get(cv2.cv.CAP_PROP_FPS)

t = 100
w = 320.0
h = 240.0

postit_img = cv2.imread('post-it4.jpg')
img_height, img_width, depth = postit_img.shape
scale_w = w / img_width
scale_h = h / img_height
postit_img = cv2.resize(postit_img, (0,0), fx=scale_w, fy=scale_h)

grey_postit_img = cv2.cvtColor(postit_img,cv2.COLOR_BGR2GRAY)

rect = np.zeros([h, w, 1], np.uint8)
rect[60:180, 40:280] = 255 * np.ones([120, 240, 1], np.uint8)

last = 0
while True:
    ret, image = capture.read()
    
    img_height, img_width, depth = image.shape
    scale_w = w / img_width
    scale_h = h / img_height
    image = cv2.resize(image, (0,0), fx=scale_w, fy=scale_h)
    image_cp = image.copy()

    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #template = grey_postit_img
    #template_corrds = np.float32([[0,0],[0,h],[w,h],[w,0] ]).reshape(-1,1,2)
    
    template = rect
    template_corrds = np.float32([[60,40],[60,280],[180,280],[180,40]]).reshape(-1,1,2)
    
    (matches, kp1_img, kp2_img, img3, M, centroid) = sift.match(template, grey_image)

    #if M is not None:
    #	dst = cv2.perspectiveTransform(template_corrds, M)
    #	img2 = cv2.polylines(grey_image,[np.int32(dst)],True,255)
    #	cv2.imshow('detected', img2)
    	
    if centroid is not None:
    	img2 = cv2.circle(image_cp, centroid, 20, (0, 0, 255), 1)
    	cv2.imshow('detected', img2) 
	
	# Compose 2x2 grid with all previews
    grid = np.zeros([2*h, 2*w, 3], np.uint8)
    grid[0:h, 0:w] = np.dstack([kp1_img])
    # We need to convert each of them to RGB from grescaled 8 bit format
    grid[0:h, w:2*w] = np.dstack([kp2_img])
    grid[h:2*h, 0:2*w] = np.dstack([img3])

    cv2.imshow('Image previews', grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
