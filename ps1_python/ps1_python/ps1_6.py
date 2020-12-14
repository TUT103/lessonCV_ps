import cv2
import numpy as np
import time
import hough_lines_acc
import hough_peaks
import hough_lines_draw
import matplotlib.pyplot as plt

font = {'family': 'SimHei',
        'color': 'black',
        'weight': 'normal',
        'size': 16,
        }


def ps1_6():
    start_time = time.time()
    img = cv2.imread("../input/ps1-input2.png")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    media_img = cv2.medianBlur(gray_img, 5)

    # 区域1
    plt.subplot(221)
    plt.title("第一次Canny的图像", fontdict=font)
    edge_img1 = cv2.Canny(media_img, 180, 230)
    plt.imshow(edge_img1, cmap='gray')

    # 区域2
    plt.subplot(222)
    plt.title("第一次查找直线", fontdict=font)
    # H1, thetas, rhos = hough_lines_acc.hough_lines_acc(edge_img1)
    # peaks1 = hough_peaks.hough_peaks(H1, 80, threshold=100)
    # green_lined1 = hough_lines_draw.hough_lines_draw(img, peaks=peaks1)
    # cv2.imwrite("../output/ps1-6-a-1.png", green_lined1)
    # green_lined1 = cv2.cvtColor(green_lined1, cv2.COLOR_BGR2RGB)
    # plt.imshow(green_lined1)

    # 区域3
    plt.subplot(223)
    plt.title("第二次Canny的图像", fontdict=font)
    edge_img2 = cv2.Canny(media_img, 50, 100)
    plt.imshow(edge_img2, cmap='gray')

    # 区域4
    plt.subplot(224)
    plt.title("第二次查找直线", fontdict=font)
    H2, thetas, rhos = hough_lines_acc.hough_lines_acc(edge_img2)
    peaks2 = hough_peaks.hough_peaks(H2, 50, threshold=150)
    print(len(peaks2))
    green_lined2 = hough_lines_draw.hough_lines_draw(img, peaks=peaks2)
    cv2.imwrite("../output/ps1-6-c-1.png", green_lined2)
    green_lined2 = cv2.cvtColor(green_lined2, cv2.COLOR_BGR2RGB)
    plt.imshow(green_lined2)

    print('6) Time elapsed: %.2f s' % (time.time() - start_time))
    plt.show()


ps1_6()
