import numpy as np
import cv2
import hough_lines_acc


def clip(idx):
    return int(max(idx, 0))


def hough_peaks(H, numpeaks=1, threshold=100, nhood_size=5):
    # TODO
    peaks = np.zeros((numpeaks, 2), dtype=np.uint64)
    temp_H = H.copy()
    for i in range(numpeaks):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp_H)  # find maximum peak
        if max_val > threshold:
            peaks[i] = max_loc
            (c, r) = max_loc
            t = nhood_size // 2.0
            temp_H[clip(r - t):int(r + t + 1), clip(c - t):int(c + t + 1)] = 0
        else:
            peaks = peaks[:i]
            break
    return peaks[:, ::-1]  # 将矩阵第二个维度逆

# test
# img = cv2.imread("../output/ps1-1-a-1.png", -1)
# H, _, _ = hough_lines_acc.hough_lines_acc(img)
# res = hough_peaks(H, 10)
# print(res)
