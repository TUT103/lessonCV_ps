import numpy as np
import cv2


def hough_lines_acc(img):
    # 1.边界检测Canny
    # 2.创建rho和thetas的范围
    thetas = np.deg2rad(np.arange(0 ,360))  # 得到长度为180,0度到180度的弧度值（0-pi）
    row, cols = img.shape
    diag_len = np.ceil(np.sqrt(row ** 2 + cols ** 2))  # 图片对角线的长度
    rho_res = np.linspace(-diag_len, diag_len, int(2 * diag_len))  # rho的大小是从-rho到rho

    cos_t = np.cos(thetas)  # cos_theta
    sin_t = np.sin(thetas)  # sin_theta
    num_theta = len(thetas)
    # 3.确定acc的范围
    accumulator = np.zeros((int(2 * diag_len), num_theta), dtype=np.uint64)  # 二维矩阵，长宽分别是2倍rho和thetas
    y_inx, x_inx = np.nonzero(img)  # 避免做无用功
    # 4.投票并寻找峰值
    for i in range(len(x_inx)):
        x = x_inx[i]
        y = y_inx[i]
        for j in range(num_theta):
            rho = round(x * cos_t[j] + y * sin_t[j])
            if isinstance(rho, int):  # 类型判断
                accumulator[rho, j] += 1
            else:
                accumulator[int(rho), j] += 1
    return accumulator, rho_res, thetas

# test
# img = cv2.imread("../output/ps1-1-a-1.png", -1)
# res, t, r = hough_lines_acc(img)
# print(res.sum())
