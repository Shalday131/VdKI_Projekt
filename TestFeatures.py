import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy

class TestFeatures:

    def __init__(self, images):
        self.images = images
        self.current_index = 0
        self.edge_image = None

    def find_circles_test(self):
        clone = copy.deepcopy(self.images[self.current_index])
        circles = cv.HoughCircles(clone, cv.HOUGH_GRADIENT_ALT, dp=1.5, minDist=25, param1=140, param2=0.5, minRadius=1, maxRadius=200)
        circles = np.uint16(np.around(circles))
        print("Kreise: ", circles)
        print("Anzahl Kreise: ", circles.size/3) # für jeden Kreis werden 3 Zahlen gespeichert --> durch 3 teilen
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
        clone = copy.deepcopy(self.images[self.current_index])
        corners = cv.goodFeaturesToTrack(clone, 1000, 0.65, 5)
        corners = np.int0(corners)
        # Bild darstellen
        for i in corners:
            x, y = i.ravel()
            cv.circle(clone, (x, y), 3, 255, -1)
        cv.imshow("Corners", clone)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_edges_test(self):
        clone = copy.deepcopy(self.images[self.current_index])
        edges = cv.Canny(clone, 100, 300)
        self.edge_image = edges
        # Bild darstellen
        plt.subplot(121), plt.imshow(clone, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def SIFT_test(self):
        clone = copy.deepcopy(self.images[self.current_index])
        sift = cv.SIFT_create()
        kp = sift.detect(clone, None)
        clone = cv.drawKeypoints(clone, kp, self.images[self.current_index])
        cv.imshow('SIFT', clone)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_contours_test(self):
        img = self.edge_image

        # find the contours
        contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # take the first contour
        cnt = contours[0]

        x_left, y_bottom, x_right, y_top = cv.boundingRect(cnt)
        x_right = 0
        y_top = 0
        # compute the bounding rectangle of the contour
        for cnt2 in contours:
            x, y, w, h = cv.boundingRect(cnt2)
            if x < x_left:    # suche kleinstes x
                x_left = x
            if y < y_bottom:    # suche kleinstes y
                y_bottom = y
            if x+w > x_right:            # suche größtes x
                x_right = x+w
            if y+h > y_top:            # suche größtes y
                y_top = y+h

        # draw contour
        img = cv.drawContours(img, [cnt], 0, (0, 255, 255), 2)

        # draw the bounding rectangle
        img = cv.rectangle(img, (x_left, y_bottom), (x_right, y_top), (255, 50, 255), 2)

        # display the image with bounding rectangle drawn on it
        cv.imshow("Bounding Rectangle", img)
        cv.waitKey(0)
        cv.destroyAllWindows()