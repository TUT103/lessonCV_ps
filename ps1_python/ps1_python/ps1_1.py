import cv2
import numpy as np
import time


def autoCanny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # 调用OpenCV的边缘函数Canny
    edged = cv2.Canny(image, lower, upper)
    return edged


def ps1_1():
    start_time = time.time()
    img = cv2.imread('../input/ps1-input0.png', cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("show", img)
    #  generate an edge image 
    edge_img = autoCanny(img, 0.5)
    # Store edge image (img_edges) as ps1-1-a-1.png
    cv2.imwrite("../output/ps1-1-a-1.png", edge_img)
    print('1) Time elapsed: %.3f s' % (time.time() - start_time))
    # cv2.waitKey()
    return edge_img

# test
# ps1_1()
