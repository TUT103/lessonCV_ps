import cv2
import numpy as np
import time

from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *
from find_circles import *
from hough_circles_draw import *
import matplotlib.pyplot as plt

font = {'family': 'SimHei',
        'color': 'black',
        'weight': 'normal',
        'size': 16,
        }


def ps1_8():
    start_time = time.time()

    img = cv2.imread("../input/ps1-input3.png")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    media_img = cv2.medianBlur(gray_img, 5)

    '''转换为俯视图的形式'''
    o = 400, 200
    x = 50
    y = 100 + 15
    min = 0
    src = np.float32([[398 - min, 272], [372 - min, 54], [592 - min, 270], [504 - min, 50]])
    dst = np.float32([[o[0] - x, o[1] + y], [o[0] - x, o[1] - y], [o[0] + x, o[1] + y], [o[0] + x, o[1] - y]])
    M1 = cv2.getPerspectiveTransform(src, dst)
    M2 = cv2.getPerspectiveTransform(dst, src)
    img_warped = cv2.warpPerspective(media_img, M1, (650, 500))
    img_warped_edge = cv2.Canny(img_warped, 30, 50)
    img_original = cv2.warpPerspective(img, M1, (650, 500))

    # plt.imshow(edge_img, cmap='gray')

    # 区域1
    plt.subplot(221)
    plt.title("将圆画在Canny后的俯视图上", fontdict=font)
    all_1 = find_circles(img_warped_edge, min_radius=15, max_radius=35, threshold=170, nhood_size=10)
    # all_1 = [[110, 437, 15], [140, 345, 15], [394, 430, 15], [402, 429, 15], [193, 382, 15], [385, 433, 15],
    #          [395, 416, 15],
    #          [376, 427, 15], [403, 420, 15], [250, 436, 15], [231, 161, 16], [111, 438, 16], [191, 383, 16],
    #          [232, 312, 16],
    #          [294, 335, 16], [259, 435, 16], [406, 435, 16], [389, 428, 16], [395, 427, 16], [251, 435, 16],
    #          [181, 269, 17],
    #          [164, 322, 17], [191, 384, 17], [252, 436, 17], [232, 160, 17], [405, 436, 17], [158, 238, 17],
    #          [379, 435, 17],
    #          [258, 438, 17], [387, 434, 17], [205, 423, 18], [254, 366, 18], [190, 385, 18], [404, 439, 18],
    #          [293, 333, 18],
    #          [169, 320, 18], [163, 323, 18], [234, 310, 18], [158, 237, 18], [257, 437, 18], [185, 268, 19],
    #          [292, 332, 19],
    #          [206, 424, 19], [253, 367, 19], [299, 328, 19], [168, 321, 19], [114, 269, 19], [162, 236, 19],
    #          [380, 414, 19],
    #          [255, 436, 19], [255, 437, 20], [184, 268, 20], [161, 237, 20], [291, 332, 20], [390, 435, 20],
    #          [298, 329, 20],
    #          [380, 415, 20], [404, 425, 20], [424, 397, 20], [263, 433, 20], [262, 433, 21], [274, 237, 21],
    #          [297, 330, 21],
    #          [390, 437, 21], [407, 425, 21], [394, 422, 21], [413, 418, 21], [383, 448, 21], [400, 432, 21],
    #          [254, 437, 21],
    #          [383, 438, 22], [407, 427, 22], [413, 419, 22], [397, 435, 22], [387, 445, 22], [391, 418, 22],
    #          [391, 437, 22],
    #          [296, 329, 22], [381, 417, 22], [421, 417, 22], [408, 427, 23], [399, 434, 23], [382, 418, 23],
    #          [385, 437, 23],
    #          [118, 267, 23], [209, 420, 23], [259, 434, 23], [295, 328, 23], [401, 420, 23], [277, 236, 24],
    #          [394, 436, 24],
    #          [386, 436, 24], [209, 418, 24], [392, 428, 24], [405, 430, 24], [389, 421, 24], [383, 418, 24],
    #          [259, 432, 24],
    #          [411, 428, 24], [392, 426, 25], [397, 438, 25], [383, 420, 25], [404, 430, 25], [386, 437, 25],
    #          [413, 432, 25],
    #          [258, 429, 26], [387, 440, 26], [387, 417, 26], [396, 439, 26], [412, 423, 26], [388, 439, 27],
    #          [397, 424, 28],
    #          [383, 423, 28], [396, 425, 29], [390, 436, 29], [389, 421, 30]]
    img_warped_edge = cv2.cvtColor(img_warped_edge, cv2.COLOR_GRAY2RGB)
    img_warped_edge_drawn_top = hough_circles_draw(img_warped_edge, all_1)
    plt.imshow(img_warped_edge_drawn_top)
    print(all_1)
    print(len(all_1))

    # 区域2
    plt.subplot(222)
    plt.title("将圆画在Canny后的斜视图上", fontdict=font)
    # list_2 = []
    # for i in range(len(all_1)):
    #     list_i = PointTransformUseM((all_1[i][0], all_1[i][1]), M2)
    #     list_i[0] = int(list_i[0])
    #     list_i[1] = int(list_i[1])
    #     list_i.append(all_1[i][2])
    #     list_2.append(list_i)
    # print(list_2)
    # drawn_img_2 = hough_circles_draw(img, list_2)
    # drawn_img_2 = cv2.cvtColor(drawn_img_2, cv2.COLOR_BGR2RGB)
    # plt.imshow(drawn_img_2)
    img_warped_edge_drawn_side = cv2.warpPerspective(img_warped_edge_drawn_top, M2, (683, 512))
    # img_warped_edge_drawn_side = cv2.cvtColor(img_warped_edge_drawn_side, cv2.COLOR_BGR2RGB)
    plt.imshow(img_warped_edge_drawn_side)

    # 区域3
    plt.subplot(223)
    plt.title("将圆画在原始图的俯视图上", fontdict=font)
    img_warped_drawn_top = hough_circles_draw(img_original, all_1)
    img_warped_drawn_top = cv2.cvtColor(img_warped_drawn_top, cv2.COLOR_BGR2RGB)
    plt.imshow(img_warped_drawn_top)

    # 区域4
    plt.subplot(224)
    plt.title("将圆画在原始图的斜视图上", fontdict=font)
    img_warped_drawn_side = cv2.warpPerspective(img_warped_drawn_top, M2, (683, 512))
    # img_warped_drawn_side = cv2.cvtColor(img_warped_drawn_side, cv2.COLOR_BGR2RGB)
    plt.imshow(img_warped_drawn_side)

    print('\033[F\r8) Time elapsed: %.2f s' % (time.time() - start_time))
    plt.show()


ps1_8()
