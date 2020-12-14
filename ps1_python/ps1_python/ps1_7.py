import cv2
import numpy as np
import time
from hough_peaks import *
from find_circles import *
from hough_circles_draw import *
import matplotlib.pyplot as plt

font = {'family': 'SimHei',
        'color': 'black',
        'weight': 'normal',
        'size': 16,
        }


def ps1_7():
    start_time = time.time()
    img = cv2.imread("../input/ps1-input2.png")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    media_img = cv2.medianBlur(gray_img, 5)

    # 区域1
    plt.subplot(221)
    plt.title("Canny后的图像", fontdict=font)
    edge_img = cv2.Canny(media_img, 50, 100)
    plt.imshow(edge_img, cmap='gray')

    # 区域2
    all_b = find_circles(edge_img, min_radius=20, max_radius=50, threshold=200, nhood_size=10)
    drawn_img_b = hough_circles_draw(img, all_b)  # 画圆
    cv2.imwrite("../output/ps1-7-a-1.png", drawn_img_b)
    print('\033[F\r7) Time elapsed: %.2f' % (time.time() - start_time))
    plt.show()


ps1_7()
