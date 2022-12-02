import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class image_preprocessing:

    def __init__(self, images):
        self.images = images

    # ändert die Größe der Bilder auf eine spezifische Größe, die eingestellt wird
    def resize(self):
        resized_images = []
        width = 700
        height = 700
        dim = (width, height)
        for image in self.images:
            resized_images.append(cv.resize(image, dim))
        self.images = resized_images

    # wandelt das Bild in ein Graustufenbild um
    def grayscale(self):
        gray_images = []
        for image in self.images:
            gray_images.append(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        self.images = gray_images

    # legt ein Unschärfefilter auf das Bild
    def blur(self):
        blurred_images = []
        for image in self.images:
            blurred_images.append(cv.GaussianBlur(image, (5, 5), 0))    # hier kann die Unschärfe geändert werden
        self.images = blurred_images

    # Unterteilt das Bild in 0- und 255-Werte nach einem bestimmten Schwellwert
    def thresholding(self):
        thresholded_images = []
        # ret, th1 = cv.threshold(self.images, 127, 255, cv.THRESH_BINARY)
        for image in self.images:
            thresholded_images.append(
                cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11,
                                     2))  # hier kann der Schwellwert bearbeitet werden
        self.images = thresholded_images

    # gibt die bearbeiteten Bilder zurück
    def get_modified_images(self):
        return self.images

    # detektiert Kreise im Bild
    def HoughCircles(self):
        circles_per_image = []
        for image in self.images:
            circles_per_image.append(len(cv.HoughCircles(image, method=cv.HOUGH_GRADIENT_ALT, dp=1.5, minDist=25, param1=140, param2=0.5, minRadius=1, maxRadius=200))) # hier können die Parameter der Kreisfindung eingestellt werden
            if circles_per_image is None:
                return
            """
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv.circle(self.images[0], (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv.circle(self.images[0], (i[0], i[1]), 2, (0, 0, 255), 3)
            cv.imshow('detected circles', self.images[0])
            cv.waitKey(0)
            cv.destroyAllWindows()
            print(circles)
            """
        return circles_per_image

    # detektiert Ecken im Bild
    def corner_tracker(self):
        corners_per_image = []
        for image in self.images:
            corners_per_image.append(len(cv.goodFeaturesToTrack(image, 1000, 0.65, 5)))
            """
            if corners is None:
                return
            corners = np.int0(corners)
            for i in corners:
                x, y = i.ravel()
                cv.circle(self.images[20], (x, y), 3, 255, -1)
            cv.imshow("Corners", self.images[20])
            cv.waitKey(0)
            cv.destroyAllWindows()
            """
        return corners_per_image