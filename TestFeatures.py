import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy

class TestFeatures:

    def __init__(self, images):
        self.images = images
        self.edge_image = None

    def find_circles_test(self):
        clone = copy.deepcopy(self.images[20])
        circles = cv.HoughCircles(clone, cv.HOUGH_GRADIENT_ALT, dp=1.5, minDist=25, param1=140, param2=0.5, minRadius=1, maxRadius=200)
        circles = np.uint16(np.around(circles))
        # Bild darstellen
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
        # Bild darstellen
        for i in corners:
            x, y = i.ravel()
            cv.circle(clone, (x, y), 3, 255, -1)
        cv.imshow("Corners", clone)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_edges(self):
        clone = copy.deepcopy(self.images[0])
        edges = cv.Canny(clone, 8, 300)
        self.edge_image = edges
        print(edges)
        # Bild darstellen
        plt.subplot(121), plt.imshow(clone, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def SIFT(self):
        clone = copy.deepcopy(self.images[20])
        sift = cv.SIFT_create()
        kp = sift.detect(clone, None)
        clone = cv.drawKeypoints(clone, kp, self.images[20])
        cv.imshow('SIFT', clone)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_contours(self):
        img = self.edge_image

        # find the contours
        contours, _ = cv.findContours(self.edge_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # take the first contour
        cnt = contours[0]
        print(cnt)
        print(contours)

        x_coordinate, y_coordinate, x2, y2 = cv.boundingRect(cnt)
        x2 = 0
        y2 = 0
        # compute the bounding rectangle of the contour
        for cnt2 in contours:
            x, y, w, h = cv.boundingRect(cnt2)
            if x < x_coordinate:
                x_coordinate = x
            if y < y_coordinate:
                y_coordinate = y
            if x+w > x2:
                x2 = x+w
            if y+h > y2:
                y2 = y+h
        print(x_coordinate, y_coordinate, x2, y2)

        # draw contour
        img = cv.drawContours(img, [cnt], 0, (0, 255, 255), 2)

        # draw the bounding rectangle
        img = cv.rectangle(img, (x_coordinate, y_coordinate), (x2, y2), (255, 255, 255), 2)

        # display the image with bounding rectangle drawn on it
        cv.imshow("Bounding Rectangle", img)
        cv.waitKey(0)
        cv.destroyAllWindows()