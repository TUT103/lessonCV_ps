import cv2
import numpy as np
import time
import hough_lines_acc
import hough_peaks
import hough_lines_draw
from ps1_1 import *


def ps1_2():
    start_time = time.time()
    # 拿到边界图像
    edge_img = ps1_1()
    # 原始图像
    ori_image_BGR = cv2.imread("../input/ps1-input0.png", 1)
    # TODO
    # a)计算累加器数组
    H, thetas, rhos = hough_lines_acc.hough_lines_acc(edge_img)
    cv2.imwrite("../output/ps1-2-a-1.png", H)
    cv2.namedWindow("hough accumulator array")
    cv2.imshow("hough accumulator array", H)
    print("图像已经出现，等待3秒自动消失或按任意键消失...")
    # cv2.waitKey(3000)
    # b)求峰值
    peaks = hough_peaks.hough_peaks(H, 10)
    print(peaks)
    peaks_x, peaks_y = H.shape[0], H.shape[1]
    peaks_mask = np.zeros((peaks_x, peaks_y))
    for i in range(len(peaks)):
        y, x = peaks[i]
        peaks_mask = cv2.circle(peaks_mask, (x, y), 5, 125, 5)
    cv2.imwrite("../output/ps1-2-b-1.png", peaks_mask)
    cv2.namedWindow("peaks image")
    cv2.imshow("peaks image", peaks_mask)
    print("图像已经出现，等待3秒自动消失或按任意键消失...")
    # cv2.waitKey(300 )
    # c)原图像画线
    hough_lines_draw.hough_lines_draw(ori_image_BGR, peaks=peaks)
    print('2) Time elapsed: %.3f s' % (time.time() - start_time))


# test
ps1_2()
