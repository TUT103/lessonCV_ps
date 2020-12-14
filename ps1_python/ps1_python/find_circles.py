import cv2
import numpy as np
from hough_circles_acc import hough_peaks_circle


# 返回一个列表，列表中的每一项包含了圆的abr
def find_circles(img, min_radius, max_radius, threshold=100, nhood_size=5):
    all_circles = []
    for each_radius in range(min_radius, max_radius):
        peaks = hough_peaks_circle(img, each_radius, threshold=threshold, nhood_size=nhood_size)
        # print(peaks)
        for each_peaks in peaks:
            all_circles.append(each_peaks)
    return all_circles


# test
# img = cv2.imread("../input/ps1-input1.png")
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# media_img = cv2.medianBlur(gray_img, 5)
# edge_media_img = cv2.Canny(media_img, 150, 150)
# cv2.imwrite("edge.png", edge_media_img)
# # hough_peaks_circle(edge_media_img, 27)
# all = find_circles(edge_media_img, min_radius=20, max_radius=25, threshold=160)
# print(all)
# print(len(all))
