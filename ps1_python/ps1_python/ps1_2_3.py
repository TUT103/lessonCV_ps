import cv2
import numpy as np
import time
import hough_lines_acc
import hough_peaks
import hough_lines_draw
import utils
from ps1_1 import *

"""
此文件为测试文件
"""


def ps1_2():
    start_time = time.time()
    # 拿到边界图像
    edge_img = ps1_1()
    # 原始图像的BGR图像
    ori_image_BGR = cv2.imread("../input/ps1-input0.png", 1)
    # TODO
    # a)计算累加器数组
    H, thetas, rhos = hough_lines_acc.hough_lines_acc(edge_img)
    utils.img_show_write(H, "hough accumulator array", "../output/ps1-2-a-1.png")
    # b)求峰值并保存峰值图片
    peaks = hough_peaks.hough_peaks(H, 10)
    # peaks = np.array([[245, 0],
    #                   [245, 90],
    #                   [127, 0],
    #                   [127, 90],
    #                   [8, 0],
    #                   [8, 90]])
    print("peaks", peaks)
    peaks_x, peaks_y = H.shape[0], H.shape[1]
    peaks_mask = np.zeros((peaks_x, peaks_y))
    for i in range(len(peaks)):
        y, x = peaks[i]
        peaks_mask = cv2.circle(peaks_mask, (x, y), 5, 125, 5)
    utils.img_show_write(peaks_mask, "../output/ps1-2-b-1.png", "peaks image")
    # c)原图像画线
    # mask = np.zeros((1000, 1000, 3))
    green_lined = hough_lines_draw.hough_lines_draw(ori_image_BGR, peaks=peaks)
    cv2.imwrite("../output/ps1-2-c-1.png", green_lined)
    print('2) Time elapsed: %.3f s' % (time.time() - start_time))
    cv2.waitKey()


# test
ps1_2()
