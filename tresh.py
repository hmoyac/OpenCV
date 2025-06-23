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

def suma():
    # print(np.array([200, 250], dtype=np.uint8).shape)
    # print(np.array([200, 250], dtype=np.uint8))
    # print(np.array([200, 250], dtype=np.uint8).reshape(-1, 1))
    arr1 = np.array([200, 250], dtype=np.uint8).reshape(-1, 1)
    arr2 = np.array([40, 40], dtype=np.uint8).reshape(-1, 1)
    add_numpy = arr1 + arr2
    add_cv2 = cv2.add(arr1, arr2)
    print(arr1)
    print(arr2)
    print(add_numpy)
    print(add_cv2)
    #[[200] [250]]+[[40 ] [40]] = [[240][ 34]]
    #[[200] [250]]cv2+[[40 ] [40]] = [[240][255]]

def video_cap():
    import cv2
    import sys

    s = 0
    if len(sys.argv) > 1:
        s = sys.argv[1]

    # name = "/home/hmoya/PycharmProjects/OpenCV/cervante/img/Cervante-logo%02d.jpg"
    # name = "cervante/Cervante-logo.jpg"
    # name = "cervante/img/Cervante-logo%02d.jpg"
    # name = "https://www.youtube.com/watch?v=wofhTAqAYpo"
    # name = "192.168.1.104"
    name = "video/modelo2.mp4"
    #source = cv2.VideoCapture("https://www.youtube.com/shorts/ogsvyRTl_6Y")
    source = cv2.VideoCapture(name)
    #
    win_name = 'Camera Preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    #
    while cv2.waitKey(1) != 27:  # Escape
        has_frame, frame = source.read()

        if not has_frame:
           print("no frame")
           break
        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)

def capture():
    import cv2

    # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    # Obtain frame size information using get() method
    #
    # 0 = cv2.CAP_PROP_POS_MSEC
    # 1 = cv2.CAP_PROP_POS_FRAMES
    # 2 = cv2.CAP_PROP_
    # 3= cv2.CAP_PROP_FRAME_WIDTH
    # 4 = cv2.CAP_PROP_FRAME_HEIGHT
    # 5 = cv2.CAP_PROP_FPS
    #
    #
    #

    # frame_width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH)) # 3
    # frame_height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 4
    # fps = vid_capture.get(cv2.CAP_PROP_FPS)  # 5
    # frame_size = (frame_width, frame_height)


    # Create a video capture object, in this case we are reading the video from a file
    # vid_capture = cv2.VideoCapture('video/modelo1.mp4')
    vid_capture = cv2.VideoCapture(0)

    frame_width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 3
    frame_height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 4
    fps = vid_capture.get(cv2.CAP_PROP_FPS)  # 5
    frame_size = (frame_width, frame_height)

    if not vid_capture.isOpened():
        print("Error opening the video file")
    # Read fps and frame count
    else:
        # Get frame rate information
        # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
        fps = vid_capture.get(cv2.CAP_PROP_FPS) # 5
        print('Frames per second : ', fps, 'FPS')

        # Get frame count
        # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
        frame_count = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT) # 7
        print('Frame count : ', frame_count)

    output = cv2.VideoWriter('output_video_from_file.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             20, frame_size)
    output2 = cv2.VideoWriter('output_video_from_file.mp4', cv2.VideoWriter_fourcc(*'XVID'),
                              20, frame_size)
    while vid_capture.isOpened():
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, frame = vid_capture.read()

        if ret:
            # if vid_capture.get(cv2.CAP_PROP_POS_FRAMES) < 2:
            #     for i in range(10):
            #         try:
            #             print(f"propiedad {i}:" + str(vid_capture.get(i)))
            #         except:
            #             break
            #     print("!#################")
            #
            #     print(f"propiedad cv2.CAP_PROP_POS_MSEC:" + str(vid_capture.get(cv2.CAP_PROP_POS_MSEC)))
            #     print(f"propiedad cv2.CAP_PROP_POS_FRAMES:" + str(vid_capture.get(cv2.CAP_PROP_POS_FRAMES)))
            #     print(f"propiedad 2:" + str(vid_capture.get(2)))
            #     print(f"propiedad cv2.CAP_PROP_FRAME_WIDTH:" + str(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            #     print(f"propiedad cv2.CAP_PROP_FRAME_HEIGHT:" + str(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            #     print(f"propiedad cv2.CAP_PROP_FPS:" + str(vid_capture.get(cv2.CAP_PROP_FPS)))
            #     print(f"propiedad 6:" + str(vid_capture.get(6)))
            #     print(f"propiedad 7:" + str(vid_capture.get(7)))
            #     print(f"propiedad 8:" + str(vid_capture.get(8)))
            #     print(f"propiedad 9:" + str(vid_capture.get(9)))

            # Obtain frame size information using get() method
            frame_width = int(vid_capture.get(3))
            frame_height = int(vid_capture.get(4))
            frame_size = (frame_width, frame_height)
            fps = 20



            # Write the frame to the output files
            output.write(frame)
            output2.write(frame)


            cv2.imshow('Frame', frame)
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv2.waitKey(20)

            if key == ord('q'):
                break
        else:
            break

    # Release the video capture object
    vid_capture.release()
    output.release()
    output2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # tresh()
    # suma()
    # video_cap()
    capture()