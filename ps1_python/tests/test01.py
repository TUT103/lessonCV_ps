# -*- coding: utf-8 -*-
import cv2
import numpy as np


def hough_detectline(img):
    # 1.边界检测Canny
    # 2.创建rho和thetas的范围
    thetas = np.deg2rad(np.arange(0, 180))# 得到长度为180,0度到180度的弧度值（0-pi）
    row, cols = img.shape
    diag_len = np.ceil(np.sqrt(row ** 2 + cols ** 2))# 图片对角线的长度
    rhos = np.linspace(-diag_len, diag_len, int(2 * diag_len)) #rho的大小是从-rho到rho

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_theta = len(thetas)
    # 3.确定acc的范围
    accumulator = np.zeros((int(2 * diag_len), num_theta), dtype=np.uint64)
    y_inx, x_inx = np.nonzero(img)
    # 4.投票并寻找峰值
    for i in range(len(x_inx)):
        x = x_inx[i]
        y = y_inx[i]
        for j in range(num_theta):
            rho = round(x * cos_t[j] + y * sin_t[j]) + diag_len
            if isinstance(rho, int):# 类型判断
                accumulator[rho, j] += 1
            else:
                accumulator[int(rho), j] += 1
    return accumulator, rhos, thetas


# image = cv2.imread(r'C:\Users\Y\Desktop\input_0.png')
# image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# image_binary=cv2.Canny(image_gray,150,255)
image = np.zeros((500, 500))
image[10:100, 10:100] = np.eye(90)
accumulator, rhos, thetas = hough_detectline(image)
# look for peaks
idx = np.argmax(accumulator)
##下面两句是寻找投票器最大值所对应的行与列，最大值对应的行就是rho的索引，对应的列就是theta的索引
# 可以用这句代替：row,col=np.unravel_index(idx,ccumulator.shape)
# rho=rho[row],theta=theta[col]
rho = rhos[int(idx / accumulator.shape[1])]
theta = thetas[idx % accumulator.shape[1]]
k = -np.cos(theta) / np.sin(theta)
b = rho / np.sin(theta)
x = np.float32(np.arange(1, 150, 2))
# 要在image 上画必须用float32，要不然会报错(float不行)
y = np.float32(k * x + b)
cv2.imshow("original image", image), cv2.waitKey(0)
for i in range(len(x) - 1):
    cv2.circle(image, (x[i], y[i]), 5, (255, 0, 0), 1)
cv2.imshow("hough", image), cv2.waitKey(0)
print("rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))
