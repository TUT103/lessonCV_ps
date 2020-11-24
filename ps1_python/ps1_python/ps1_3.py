import cv2
import numpy as np
import time
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *
import matplotlib.pyplot as plt


def ps1_3():
    start_time = time.time()
    #  3a: smooth the noisy image using gaussian blurring
    noise_img = cv2.imread("../input/ps1-input0-noise.png", 0)
    gaussian_noise_img = cv2.GaussianBlur(noise_img, (13, 11), 0, 0)
    plt.subplot(231)
    plt.imshow(gaussian_noise_img, cmap="Greys_r")
    cv2.imwrite("../output/ps1-3-a-1.png", gaussian_noise_img)
    plt.title("3a1")

    #  3b: perform edge detection on both images using Canny
    edge_noise_img = cv2.Canny(noise_img, 150, 150)
    edge_gaussian_noise_img = cv2.Canny(gaussian_noise_img, 70, 70)
    plt.subplot(232)
    plt.imshow(edge_noise_img, cmap="Greys_r")
    cv2.imwrite("../output/ps1-3-b-1.png", edge_noise_img)
    plt.title("3b1")
    plt.subplot(233)
    plt.imshow(edge_gaussian_noise_img, cmap="Greys_r")
    cv2.imwrite("../output/ps1-3-b-2.png", edge_gaussian_noise_img)
    plt.title("3b2")

    #  3c: apply hough line detection to the smoothed image
    plt.subplot(234)
    H, thetas, rhos = hough_lines_acc.hough_lines_acc(edge_gaussian_noise_img)
    peaks = hough_peaks(H, 10, threshold=75)
    print("计算出的peaks值", peaks)
    peaks_x, peaks_y = H.shape[0], H.shape[1]
    peaks_mask = np.zeros((peaks_x, peaks_y))
    for i in range(len(peaks)):
        y, x = peaks[i]
        peaks_mask = cv2.circle(peaks_mask, (x, y), 5, 125, 5)
    plt.imshow(peaks_mask, cmap="Greys_r")
    cv2.imwrite("../output/ps1-3-c-1.png", peaks_mask)
    plt.title("3c1")
    # c)原图像画线
    plt.subplot(235)
        # 原始图像的BGR图像
    ori_image_BGR = cv2.imread("../input/ps1-input0.png", 1)
    green_lined = hough_lines_draw(ori_image_BGR, peaks=peaks)
    plt.imshow(green_lined, cmap="Greys_r")
    cv2.imwrite("../output/ps1-3-c-2.png", green_lined)
    plt.title("3c2")
    plt.subplot(236)
        # 在mask上画线
    mask = np.zeros((1000, 1000, 3))
    green_lined_mask = hough_lines_draw(mask, peaks=peaks)
    plt.imshow(green_lined_mask, cmap="Greys_r")
    plt.title("green_lined_mask")
    # save the produced images
    print('3) Time elapsed: %.3f s' % (time.time() - start_time))
    plt.savefig("../output/myOutPut/ps1-3.png")
    plt.show()

# test
# ps1_3()
