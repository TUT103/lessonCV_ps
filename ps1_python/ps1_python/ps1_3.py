import cv2
import numpy as np
import time
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *

def ps1_3():
    start_time = time.time()
    
    #  3a: smooth the noisy image using gaussian blurring
    
    #  3b: perform edge detection on both images using Canny

    #  3c: apply hough line detection to the smoothed image

    #  save the produced images


    print('3) Time elapsed: %.2f s'%(time.time()-start_time))
