# import the necessary packages
#import imutils
import csv
import cv2

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		h=int(image.shape[0] / scale)
		dim = (w, h)
		image = cv2.resize(image, dim)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	i = 0
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
			row = [x, y, image[y:y + windowSize[1], x:x + windowSize[0]]]
			print(row)
			cv2.imshow('img',image[y:y + windowSize[1], x:x + windowSize[0]])
			cv2.imwrite('pic{:>05}.jpg'.format(i), image[y:y + windowSize[1], x:x + windowSize[0]])
			i+=1
			with open('vectors.csv', 'a') as csvFile:
				writer = csv.writer(csvFile)
				writer.writerow(row)
			csvFile.close()

