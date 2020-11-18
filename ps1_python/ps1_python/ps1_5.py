import cv2
import numpy as np
import time
from hough_peaks import *
from find_circles import *
from hough_circles_draw import *


def ps1_5():
    start_time = time.time()
    # 5a: Load coin image, smooth, detect edges and calculate hough space
    
    # detect circles with radius = 20 and save image
    
    # 5b: detect circles in the range [20 50]
    
    print('\033[F\r5) Time elapsed: %.2f s'%(time.time()-start_time))
