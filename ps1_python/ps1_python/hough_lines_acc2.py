import numpy as np
import cv2
import ps1_1

img = ps1_1.ps1_1()


def hough_lines_acc(img, rho_res=1, thetas=np.arange(-90, 90, 1)):
    # TODO
    # r1 = cv2.Canny(img, 50, 100)
    accumulator = cv2.HoughLines(img, 1, np.pi / 180, 200)
    # print(accumulator)
    accSize = accumulator.shape
    rhos = accSize[0]
    thetas = accSize[2]

    # 创建画布
    mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.putText(mask, "hough accumulator array:", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    for i in range(rhos):
        cv2.putText(mask, str(accumulator[i][0]), (5, i * 15 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
    cv2.imwrite("../output/ps1-2-a-1.png", mask)
    cv2.imshow("mask", mask)
    cv2.waitKey()
    return accumulator, thetas, rhos


hough_lines_acc(img)
