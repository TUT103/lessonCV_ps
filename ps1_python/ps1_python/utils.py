import cv2


def img_show_write(img, fileout, windowName):
    cv2.namedWindow(windowName)
    cv2.imshow(str(img), img)
    cv2.imwrite(str(fileout), img)
    print(str(img) + "图像已经出现，等待3秒自动消失或按任意键消失...")
