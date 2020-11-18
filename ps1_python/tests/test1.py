import cv2 as cv

image = cv.imread("../input/ps1-input0.png", 0)

r1 = cv.Canny(image, 1, 50)
r2 = cv.Canny(image, 50, 100)
r3 = cv.Canny(image, 100, 150)
r4 = cv.Canny(image, 150, 200)
cv.imshow("img", image)
cv.imshow("r1", r1)
cv.imshow("r2", r2)
cv.imshow("r3", r3)
cv.imshow("r4", r4)
cv.waitKey()
