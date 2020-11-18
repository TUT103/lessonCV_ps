import cv2
import numpy as np
import time
from hough_lines_acc import *
from hough_peaks import *
from hough_lines_draw import *


def ps1_4():
    start_time = time.time()
    #  4a: Load the coin image, smooth it (gaussian blur) and save it
    
    #  4b: apply edge detection using Canny
   
    #  4c: apply hough line detection to the smoothed image
  
    print('4) Time elapsed: %.2f s'%(time.time()-start_time))
