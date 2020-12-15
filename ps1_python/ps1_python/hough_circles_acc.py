import cv2
import numpy as np
from hough_peaks import *
import matplotlib.pyplot as plt


# 查找圆，返回acc
def hough_circles_acc(edge_img, radius):
    accumulator = np.zeros(edge_img.shape, dtype=np.uint8)
    yis, xis = np.nonzero(edge_img)  # coordinates of edges
    (m, n) = edge_img.shape
    for x, y in zip(xis, yis):
        theta = np.arange(0, 360)
        a = (y - radius * np.sin(theta * np.pi / 180)).astype(np.uint)
        b = (x - radius * np.cos(theta * np.pi / 180)).astype(np.uint)
        valid_idxs = np.nonzero((a < m) & (b < n))
        a, b = a[valid_idxs], b[valid_idxs]
        c = np.stack([a, b], 1)
        cc = np.ascontiguousarray(c).view(np.dtype((np.void, c.dtype.itemsize * c.shape[1])))
        _, idxs, counts = np.unique(cc, return_index=True, return_counts=True)
        uc = c[idxs]
        accumulator[uc[:, 0], uc[:, 1]] += counts.astype(np.uint)
    return accumulator


# 查找指定半径圆的圆心坐标
def hough_peaks_circle(img, radius, threshold, nhood_size=5):
    acc = hough_circles_acc(img, radius)
    peaks = hough_peaks(acc, numpeaks=10, threshold=threshold, nhood_size=nhood_size)
    print("半径：", radius, "，peaks：", peaks)
    peaks = peaks.tolist()  # 将NumPy形式转换为list
    for each in peaks:  # 为每一个圆心坐标加上半径
        each.append(radius)
    return peaks

# test
# img = cv2.imread("../input/ps1-input1.png")
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# media_img = cv2.medianBlur(gray_img, 5)
# edge_media_img = cv2.Canny(media_img, 150, 150)
# cv2.imwrite("edge.png", edge_media_img)
# # hough_peaks_circle(edge_media_img, 27)
# all = find_circles10(edge_media_img, min_radius=20, max_radius=25)
# print(all)
# print(len(all))
