import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy

class TestFeatures:

    def __init__(self, images):
        self.images = images

    def find_circles_test(self):
        clone = copy.deepcopy(self.images[20])
        circles = cv.HoughCircles(clone, cv.HOUGH_GRADIENT_ALT, dp=1.5, minDist=25, param1=140, param2=0.5, minRadius=1, maxRadius=200)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(clone, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(clone, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv.imshow('detected circles', clone)
        cv.waitKey(0)
        cv.destroyAllWindows()


    def find_corners_test(self):
        clone = copy.deepcopy(self.images[20])
        corners = cv.goodFeaturesToTrack(clone, 1000, 0.65, 5)
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv.circle(clone, (x, y), 3, 255, -1)
        cv.imshow("Corners", clone)
        cv.waitKey(0)
        cv.destroyAllWindows()