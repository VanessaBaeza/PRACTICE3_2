# USAGE
# python sliding_window.py --image images/adrian_florida.jpg

# import the necessary packages
from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
import argparse
import time
import cv2
import numpy as np

# load the image and define the window width and height
image = cv2.imread('caja_x.jpg')
#image = cv2.imread(args["image"])
(winW, winH) = (128, 128) 

min_match=10
detect=cv2.xfeatures2d.SIFT_create()
detector=cv2.AKAZE_create()

flann_index_kdittre=0
flann_param=dict(algorithm=flann_index_kdittre, tree=5)
search_param=dict(checks=50)
flann=cv2.FlannBasedMatcher(flann_param,search_param)

#Imagen del objeto a detectar
train_img=cv2.imread('caja.jpg',0)
(trainKP,trainDesc)=detector.detectAndCompute(train_img,None)

(queryKP,queryDesc)=detector.detectAndCompute(image,None)
matches=flann.knnMatch(np.asarray(queryDesc,np.float32),np.asarray(trainDesc,np.float32),2)

# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
		# csvData=sliding_window(resized, stepSize=32, windowSize=(winW, winH))
		# cvsFile='vectors.cvs'
		# with open('vectors.csv', 'w') as csvFile:
		# 	writer = csv.writer(csvFile)
		# 	writer.writerows(csvData)
		# csvFile.close()
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		#(queryKP,queryDesc)=detector.detectAndCompute(image,None)
		#matches=flann.knnMatch(np.asarray(queryDesc,np.float32),np.asarray(trainDesc,np.float32),2)

		goodMatch=[]
		for m,n in matches:
			if(m.distance < 0.7*n.distance):
				goodMatch.append(m)
		print(len(goodMatch))
		if (len(goodMatch)>min_match):
			tp=[]
			qp=[]
			for m in goodMatch:
				tp.append(trainKP[m.trainIdx].pt)
				qp.append(queryKP[m.queryIdx].pt)
			tp,qp=np.float32((tp,qp))
			H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
			h,w=train_img.shape
			trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
			queryBorder=cv2.perspectiveTransform(trainBorder,H)
			cv2.polylines(image,[np.int32(queryBorder)],True,(0,255,0),5)
		else:
			print("no matches")

		#cv2.imshow('result',image)

		# since we do not have a classifier, we'll just draw the window
		clone = resized.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)
		cv2.waitKey(1)
		time.sleep(0.025)
