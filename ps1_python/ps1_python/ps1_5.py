import cv2
import numpy as np
import time
from hough_peaks import *
from find_circles import *
from hough_circles_draw import *
import matplotlib.pyplot as plt


def ps1_5():
    start_time = time.time()
    # 5a: Load coin image, smooth, detect edges and calculate hough space
    img = cv2.imread("../input/ps1-input1.png")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smoothed_img = cv2.GaussianBlur(gray_img, (9, 9), 3)  # smooth image
    edge_img = cv2.Canny(smoothed_img, 30, 50)
    cv2.imwrite("../output/ps1-5-a-1.png", smoothed_img)
    cv2.imwrite("../output/ps1-5-a-2.png", edge_img)

    # detect circles with radius = 20 and save image
    all_a = hough_peaks_circle(edge_img, radius=20, threshold=140, nhood_size=12)
    drawn_img_a = hough_circles_draw(img, all_a)   # 画圆
    cv2.imwrite("../output/ps1-5-a-3.png", drawn_img_a)

    # 5b: detect circles in the range [20 50]
    all_b = find_circles(edge_img, min_radius=20, max_radius=30, threshold=180, nhood_size=12)
    drawn_img_b = hough_circles_draw(img, all_b)  # 画圆
    cv2.imwrite("../output/ps1-5-b-1.png", drawn_img_b)
    print('\033[F\r5) Time elapsed: %.2f s' % (time.time() - start_time))


ps1_5()