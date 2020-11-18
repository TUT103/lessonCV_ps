import numpy as np
import cv2
import hough_lines_acc


def clip(idx):
    return int(max(idx, 0))


# accumulator, thetas, rhos = hough_lines_acc.hough_lines_acc()


def hough_peaks(H=None, numpeaks=1, threshold=100, nhood_size=5):
    # TODO
    accumulator, thetas, rhos = hough_lines_acc.hough_lines_acc(
        cv2.imread('../output/ps1-1-a-1.png', -1))
    # 创建空白画布
    w, h = 256, 256
    mask = np.zeros((w, h), dtype=np.uint8)
    for line in accumulator:
        r, t = line[0]
        a = np.cos(t)
        b = np.sin(t)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 100 * (-b))
        y1 = int(y0 + 100 * a)
        x2 = int(x0 - 100 * (-b))
        y2 = int(y0 - 100 * a)
        cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
    cv2.imshow("mask", mask)
    cv2.imwrite("../output/ps1-2-b-1.png", mask)
    # cv2.imwrite("../output/ps1-2-a-1.png", mask)
    cv2.waitKey()

# return peaks[:, ::-1]

hough_peaks()
