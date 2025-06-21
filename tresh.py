from datetime import datetime
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import os
import cv2

def tresh():
    # Read the original image
    #img_read = cv2.imread("Piano_Sheet_Music.png", cv2.IMREAD_GRAYSCALE)
    img_read = cv2.imread(os.path.join(os.getcwd(), "thresholding_image/input_image.jpg"), cv2.IMREAD_GRAYSCALE)
    #img_read = cv2.imread("building-windows.jpg", cv2.IMREAD_GRAYSCALE)
    #img_read = cv2.imread("New_Zealand_Boat.jpg", cv2.IMREAD_GRAYSCALE)

    #img_read = cv2.imread(os.path.join(os.getcwd(), "thresholding_image/lucia.jpg"), cv2.IMREAD_GRAYSCALE)
    '''
        t11=datetime.now()
        # Perform global thresholding
        retval, img_thresh_gbl_1 = cv2.threshold(img_read, 50, 255, cv2.THRESH_BINARY)
        t12=datetime.now()
        tt1=t12-t11
    
        t21=datetime.now()
        # Perform global thresholding
        retval, img_thresh_gbl_2 = cv2.threshold(img_read, 130, 255, cv2.THRESH_BINARY)
        t22=datetime.now()
        tt2=t22-t21
    '''

    t31=datetime.now()
    # Perform adaptive thresholding
    img_thresh_adp = cv2.adaptiveThreshold(img_read, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)
    t32=datetime.now()
    tt3=t32-t31

    t41=datetime.now()
    # Perform adaptive thresholding
    img_thresh_adp_g = cv2.adaptiveThreshold(img_read, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
    t42=datetime.now()
    tt4=t42-t41

    #pl([[img_read, "Original"], [img_thresh_gbl_1,"Thresholded (global: 50)"], [img_thresh_gbl_2,"Thresholded (global: 130)"], [img_thresh_adp,"Thresholded (adaptive)"]],2,2,"grey")
    # Show the images
    plt.figure()
    #plt.subplot(321); plt.imshow(img_read,        cmap="gray");  plt.title("Original ");
    plt.subplot(121); plt.imshow(img_thresh_adp,cmap="gray");  plt.title("Thresholded (adaptive) "+str(tt3.microseconds));
    #plt.subplot(323); plt.imshow(img_thresh_gbl_2,cmap="gray");  plt.title("Thresholded (global: 130) "+str(tt2.microseconds));
    #plt.subplot(324); plt.imshow(img_thresh_gbl_1,  cmap="gray");  plt.title("Thresholded (global: 50) "+str(tt3.microseconds));
    plt.subplot(122); plt.imshow(img_thresh_adp_g,  cmap="gray");  plt.title("Thresholded (adaptive) Gausian "+str(tt4.microseconds));
    plt.show()

if __name__ == "__main__":
    tresh()