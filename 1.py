import cv2 as cv
import numpy as np

path = 'dataset/'
img = cv.imread(path + 'IMG.JPG', cv.IMREAD_GRAYSCALE)
first_image = img.copy()

kernel = np.ones((7, 7), np.uint8)
img = cv.GaussianBlur(img, (3, 3), 3)
img = cv.Canny(img, 70, 100)
img = cv.dilate(img, kernel, 5)
img = cv.resize(img, (800, 1200))
img = cv.bitwise_not(img)
cv.putText(img, " wor", (0, 500), cv.FONT_HERSHEY_SIMPLEX, 2, (0), 3)

print(img)
cv.imshow('res', img)
cv.waitKey(0)