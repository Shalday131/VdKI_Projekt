import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class ImagePreprocessing:

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
        for image in self.images:
            thresholded_images.append(cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2))  # hier kann der Schwellwert bearbeitet werden
        self.images = thresholded_images

    # gibt die bearbeiteten Bilder zurück
    def get_modified_images(self):
        return self.images

    # detektiert Kreise im Bild
    def find_circles(self):
        num_circles_per_image = []
        circles_per_image = []
        for image in self.images:
            circles_per_image = cv.HoughCircles(image, method=cv.HOUGH_GRADIENT_ALT, dp=1.5, minDist=25, param1=140, param2=0.5, minRadius=1, maxRadius=200) # hier können die Parameter der Kreisfindung eingestellt werden
            circles_per_image = np.uint16(np.around(circles_per_image))
            if circles_per_image is None:
                return
            else:
                circles_per_image = circles_per_image.size/3
            num_circles_per_image.append(circles_per_image)
        print(num_circles_per_image)
        return num_circles_per_image

    # detektiert Ecken im Bild
    def find_corners(self):
        corners_per_image = []
        for image in self.images:
            corners_per_image.append(len(cv.goodFeaturesToTrack(image, 1000, 0.65, 5)))
        return corners_per_image

    # findet das kleinste Rechteck, dass das Objekt umgibt
    def find_contours(self):
        aspect_ratios = []
        for image in self.images:
            edges = cv.Canny(image, 100, 300)   # hebt die Umrisse des Objekts auf dem Bild hervor
            contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # finde alle Konturen des Objekts im Bild

            # take the first contour
            cnt = contours[0]

            x_left, y_bottom, x_right, y_top = cv.boundingRect(cnt)
            x_right = 0
            y_top = 0
            for cnt2 in contours:   # berechnet das kleinste Rechteck, dass das Objekt im Bild umgibt
                x, y, w, h = cv.boundingRect(cnt2)
                if x < x_left:    # suche kleinstes x
                    x_left = x
                if y < y_bottom:    # suche kleinstes y
                    y_bottom = y
                if x+w > x_right:            # suche größtes x
                    x_right = x+w
                if y+h > y_top:            # suche größtes y
                    y_top = y+h
            width = x_right - x_left
            height = y_top - y_bottom
            current_aspect_ratio = width/height
            aspect_ratios.append(current_aspect_ratio)
        return aspect_ratios