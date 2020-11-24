import cv2
import numpy as np
from matplotlib import pyplot as plt


# def hough_lines_draw(img, peaks):
#     if len(img.shape) == 2:
#         row, col = img.shape
#     else:
#         row, col, _ = img.shape
#
#     for (rho, theta) in peaks:
#         # print(rho, theta)
#         sin_theta = np.sin(np.deg2rad(theta))
#         cos_theta = np.cos(np.deg2rad(theta))
#         # for i in range(len(peaks)):
#         print(-cos_theta / sin_theta, rho / sin_theta)
#         # for x in range(col):
#         #     y = -x * cos_theta / sin_theta + rho / sin_theta
#             # cv2.circle(img, (x, int(y)), 2, (0, 255, 0), 2)
#
#     cv2.imwrite("../output/ps1-2-c-1.png", img)
#     return img
def hough_lines_draw(img, peaks, rhos=None, thetas=None):
    plt.figure(2)
    row = 0
    col = 0
    if len(img.shape) == 2:
        row, col = img.shape
    else:
        row, col, _ = img.shape

    plt.imshow(img, cmap="Greys_r")
    for (d, t) in peaks:
        if np.fabs(np.sin(np.deg2rad(t))) > 0.1:
            x = []
            y = []
            for xx in range(col):
                yy = 1 / np.sin(np.deg2rad(t)) * (np.cos(np.deg2rad(t)) * xx - (d - 2 * max(row, col)))
                if yy >= 0 and yy <= row:
                    x.append(xx)
                    y.append(yy)
            plt.plot(x, y, color="red")
        else:
            plt.vlines(abs(d - 2 * max(row, col)), ymin=0, ymax=row, color="red")


    plt.show()
    return img
