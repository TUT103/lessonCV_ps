import cv2
import numpy as np


def hough_circles_draw(img, all_circles ):
    for each_circle in all_circles:
        img = cv2.circle(img,
                         (each_circle[0], each_circle[1]),
                         each_circle[2],
                         (0, 255, 0),
                         1)
    return img
