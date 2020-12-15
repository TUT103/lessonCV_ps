import cv2
import numpy as np
import time
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *
from ps1_1 import *
import matplotlib.pyplot as plt


def ps1_2():
    start_time = time.time()
    # 拿到边界图像
    edge_img = ps1_1()
    # a)计算累加器数组
    H, thetas, rhos = hough_lines_acc.hough_lines_acc(edge_img)

    plt.subplot(131)
    plt.imshow(H, cmap="Greys_r")
    plt.title("2a1")
    cv2.imwrite("../output/ps1-2-a-1.png", H)
    # b)求峰值并保存峰值图片
    peaks = hough_peaks(H, 10, threshold=220)
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

    plt.subplot(132)
    plt.imshow(peaks_mask, cmap="Greys_r")
    plt.title("2b1")
    cv2.imwrite("../output/ps1-2-b-1.png", peaks_mask)
    # c)原图像画线
    ori_image_BGR = cv2.imread("../input/ps1-input0.png", 1)  # 原始图像的BGR图像
    green_lined = hough_lines_draw(ori_image_BGR, peaks=peaks)

    plt.subplot(133)
    plt.imshow(green_lined, cmap="Greys_r")
    plt.title("2c1")
    cv2.imwrite("../output/ps1-2-c-1.png", green_lined)
    print('2) Time elapsed: %.3f s' % (time.time() - start_time))
    plt.savefig("../output/myOutPut/ps1-2.png")
    plt.show()


# test
ps1_2()
