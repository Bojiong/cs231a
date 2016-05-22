import cv2
import time
import math
import numpy as np

def match(img1, img2):
	sift = cv2.xfeatures2d.SIFT_create()
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp1_img = cv2.drawKeypoints(img1, kp1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
	kp2, des2 = sift.detectAndCompute(img2,None)
	kp2_img=cv2.drawKeypoints(img2,kp2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)
	
	good = []
	src = []
	dst = []
	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append([m])
	
	src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
	
	des_cor = [kp2[m[0].trainIdx].pt for m in good]
	dst_pts = np.float32(des_cor).reshape(-1,1,2)
	
	centroid = findCentroid(des_cor)
	
	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, img2, flags=2)
	
	M, mask = cv2.findHomography(src_pts, dst_pts)
	
	
	return (matches, kp1_img, kp2_img, img3, M, centroid)


def findCentroid(des_cor):
	des_len = len(des_cor)
	if des_len > 0:
		return (int(sum([x for x, y in des_cor]) / des_len), int(sum([y for x, y in des_cor]) / des_len))
	else:
		return None
