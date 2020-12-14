import cv2
import numpy as np
import time
# from hough_lines_acc import *
# from hough_peaks import *
# from hough_lines_draw import *
import hough_peaks
import hough_lines_acc
import hough_lines_draw
import matplotlib.pyplot as plt


def ps1_4():
    start_time = time.time()
    #  4a: Load the coin image, smooth it (gaussian blur) and save it
    plt.subplot(221)
    ori_img_gray = cv2.imread("../input/ps1-input1.png", 0)
    gaussian_img_grey = cv2.GaussianBlur(ori_img_gray, (5, 5), 0, 0)
    plt.imshow(gaussian_img_grey, cmap="Greys_r")
    cv2.imwrite("../output/ps1-4-a-1.png", gaussian_img_grey)
    plt.title("4a1")

    #  4b: apply edge detection using Canny
    edge_gaussian_img = cv2.Canny(gaussian_img_grey, 150, 150)
    plt.subplot(222)
    plt.imshow(edge_gaussian_img, cmap="Greys_r")
    cv2.imwrite("../output/ps1-4-b-1.png", edge_gaussian_img)
    plt.title("4b1")

    #  4c: apply hough line detection to the smoothed image
    # 高亮显示H
    plt.subplot(223)
    H, thetas, rhos = hough_lines_acc.hough_lines_acc(edge_gaussian_img)
    peaks = hough_peaks.hough_peaks(H, 10, 120)
    print("ps1-4计算出的peaks值", peaks)
    peaks_x, peaks_y = H.shape[0], H.shape[1]
    peaks_mask = np.zeros((peaks_x, peaks_y))
    for i in range(len(peaks)):
        y, x = peaks[i]
        peaks_mask = cv2.circle(peaks_mask, (x, y), 5, 125, 5)
    plt.imshow(peaks_mask, cmap="Greys_r")
    cv2.imwrite("../output/ps1-4-c-1.png", peaks_mask)

    # 显示出笔的轮廓
    plt.subplot(224)
    # mask = np.zeros((1000, 1000, 3))
    gaussian_img_RGB = cv2.cvtColor(gaussian_img_grey, cv2.COLOR_GRAY2RGB)
    green_lined = hough_lines_draw.hough_lines_draw(gaussian_img_RGB, peaks)
    cv2.imwrite("../output/ps1-4-c-2.png", green_lined)
    plt.imshow(green_lined, cmap="Greys_r")
    plt.title("4c2")
    print('4) Time elapsed: %.3f s' % (time.time() - start_time))
    plt.savefig("../output/myOutPut/ps1-5.png")
    plt.show()


# test
# ps1_4()
