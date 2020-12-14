import cv2
import numpy as np


def hough_circles_acc(edge_img, radius, max_point=10):
    thetas = np.deg2rad(np.arange(0, 360))
    num_theta = len(thetas)
    height, width = edge_img.shape[0:2]
    print(height, width)
    cos_t = np.cos(thetas)  # cos_theta
    sin_t = np.sin(thetas)  # sin_theta
    accumulator = np.zeros((height, width), dtype="uint64")
    for y in range(radius + 2, height - radius - 2):
        print(y)
        for x in range(radius + 2, width - radius - 2):
            for i in range(num_theta):
                if i % 10 == 0:
                    # 获取圆的边缘坐标x0,y0
                    if isinstance(round(y + radius * sin_t[i]), int):
                        y0 = round(y + radius * sin_t[i])
                    else:
                        y0 = int(round(y + radius * sin_t[i]))
                    if isinstance(round(x + radius * cos_t[i]), int):
                        x0 = round(x + radius * cos_t[i])
                    else:
                        x0 = int(round(x + radius * cos_t[i]))
                    # 判断是否构成了圆
                    if not edge_img[y0, x0] == 0:
                        accumulator[y, x] += 1
                        if accumulator[y, x] == max_point:
                            break
    return accumulator

# img = cv2.imread("../output/ps1-4-b-1.png")
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# acc = hough_circles_acc(img, 20, max_point=5)
# print(acc.sum())
# acc = acc.astype("uint8")
# print(acc.sum())
# cv2.imwrite("acc.png", acc)
# print(acc)
