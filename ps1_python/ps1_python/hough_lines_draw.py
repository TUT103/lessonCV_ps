import cv2
import numpy as np


def hough_lines_draw(img, peaks):
    if len(img.shape) == 2:
        row, col = img.shape
    else:
        row, col, _ = img.shape
    for (rho, theta) in peaks:
        sin_theta = np.sin(np.deg2rad(theta))
        cos_theta = np.cos(np.deg2rad(theta))
        if np.sin(theta) == 0:
            for y in range(row):
                cv2.circle(img, (int(round(rho / np.cos(theta))), y), 2, (0, 255, 0), 1)
        else:
            for x in range(col):
                y = -x * cos_theta / sin_theta + rho / sin_theta
                if not 0 <= y <= row:
                    continue
                cv2.circle(img, (x, int(round(y))), 2, (0, 255, 0), 1)
    return img
