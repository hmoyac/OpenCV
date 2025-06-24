import cv2
import sys
import numpy as np

PREVIEW  = 0  # Preview Mode
BLUR     = 1  # Blurring Filter
FEATURES = 2  # Corner Feature Detector
CANNY    = 3  # Canny Edge Detector
FLIP     = 4  # Flip image
FILTER   = 5  # Filter image cv2.filter2D

feature_params = dict(
                        maxCorners=500,
                        qualityLevel=0.2,
                        minDistance=15,
                        blockSize=9)
# s = 0
s = "images/video/modelo0.mp4"
if len(sys.argv) > 1:
    s = sys.argv[1]

image_filter = PREVIEW
alive = True

win_name = "Camera Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

source = cv2.VideoCapture(s)

while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 110, 150)
    elif image_filter == FLIP:
        result = cv2.flip(frame, 1)
    elif image_filter == FILTER:
        kernel = np.ones([5,5], np.float32) / 20
        result = cv2.filter2D(frame,-1,kernel)
    elif image_filter == BLUR:
        result = cv2.blur(frame, (10, 10))
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:
            for x, y in np.int32(corners).reshape(-1, 2):
                cv2.circle(result, (x, y), 10, (0, 255, 0), 1)

    cv2.imshow(win_name, result)

    key = cv2.waitKey(1)
    if key == ord("Q") or key == ord("q"): #or key == 27:
        alive = False
    elif key == ord("C") or key == ord("c"):
        image_filter = CANNY
    elif key == ord("B") or key == ord("b"):
        image_filter = BLUR
    elif key == ord("E") or key == ord("e"):
        image_filter = FEATURES
    elif key == ord("P") or key == ord("p"):
        image_filter = PREVIEW
    elif key == ord("F") or key == ord("f"):
        image_filter = FLIP
    elif key == ord("S") or key == ord("s"):
        image_filter = FILTER

source.release()
cv2.destroyWindow(win_name)