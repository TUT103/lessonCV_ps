import cv2
import numpy as np
import time


def ps1_1():
    start_time = time.time()
    img = cv2.imread('../input/ps1-input0.png', cv2.IMREAD_GRAYSCALE)
    #  generate an edge image 
    r1 = cv2.Canny(img, 50, 100)
    #  Store edge image (img_edges) as ps1-1-a-1.png
    cv2.imwrite("../output/ps1-1-a-1.png", r1)
    print('1) Time elapsed: %.3f s' % (time.time() - start_time))

# test
ps1_1()
