import cv2
import numpy as np
import time
import hough_lines_acc
import hough_peaks
import hough_lines_draw
from ps1_1 import *
import math

"""
此文件为测试文件
"""


def ps1_2():
    start_time = time.time()
    # 拿到边界图像
    edge_img = ps1_1()
    edge_row, edge_col = edge_img.shape
    """将edge_img旋转30度"""
    M = cv2.getRotationMatrix2D((edge_col / 2, edge_row / 2), 30, 1 / math.sqrt(2))
    rotate_edge = cv2.warpAffine(edge_img, M, (edge_col, edge_row))
    cv2.imshow("rotate", rotate_edge)
    # 原始图像的BGR图像
    ori_image_BGR = cv2.imread("../input/ps1-input0.png", 1)
    # TODO
    # a)计算累加器数组
    H, thetas, rhos = hough_lines_acc.hough_lines_acc(rotate_edge)
    cv2.imwrite("../output/ps1-2-a-1.png", H)
    cv2.namedWindow("hough accumulator array")
    cv2.imshow("hough accumulator array", H)
    print("图像已经出现，等待3秒自动消失或按任意键消失...")
    # b)求峰值并保存峰值图片
    peaks = hough_peaks.hough_peaks(H, 10)
    print("peaks", peaks)
    peaks = np.array([[267, 90], [757, 0], [385, 90], [639, 0], [504, 90], [520, 0]])
    peaks_x, peaks_y = H.shape[0], H.shape[1]
    peaks_mask = np.zeros((peaks_x, peaks_y))
    for i in range(len(peaks)):
        y, x = peaks[i]
        peaks_mask = cv2.circle(peaks_mask, (x, y), 5, 125, 5)
    cv2.imwrite("../output/ps1-2-b-1.png", peaks_mask)
    cv2.namedWindow("peaks image")
    cv2.imshow("peaks image", peaks_mask)
    print("图像已经出现，等待3秒自动消失或按任意键消失...")
    # c)原图像画线
    # mask = np.zeros((1000, 1000, 3))
    rotate_ori_BGR = cv2.warpAffine(ori_image_BGR, M, (edge_col, edge_row))
    green_line_rotate = hough_lines_draw.hough_lines_draw(rotate_ori_BGR, peaks=peaks)
    MM = cv2.getRotationMatrix2D((edge_col / 2, edge_row / 2), -30, math.sqrt(2))
    green_line = cv2.warpAffine(green_line_rotate, MM, (edge_col, edge_row))
    cv2.imwrite("../output/ps1-2-c-1.png", green_line)
    print('2) Time elapsed: %.3f s' % (time.time() - start_time))
    cv2.waitKey()


# test
ps1_2()
