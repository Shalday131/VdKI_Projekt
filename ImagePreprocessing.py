import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class ImagePreprocessing:

    def __init__(self, images):
        self.images = images
        # Eckpunkte für die jeweiligen Rechtecke um die Flasche herum abspeichern
        self.x_left = []
        self.x_right = []
        self.y_bottom = []
        self.y_top = []

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
        image_index = 0
        for image in self.images:
            circles_per_image = cv.HoughCircles(image, method=cv.HOUGH_GRADIENT_ALT, dp=1.5, minDist=25, param1=140, param2=0.5, minRadius=1, maxRadius=200) # hier können die Parameter der Kreisfindung eingestellt werden
            circles_per_image = np.uint16(np.around(circles_per_image))
            circles_per_image = circles_per_image[0]
            if circles_per_image is None:
                num_circles_per_image.append(0)
            else:
                circle_counter = 0
                for circle in circles_per_image:
                    if circle[1] <= (self.y_top[image_index]+self.y_bottom[image_index])/2:
                        if circle[1] >= self.y_bottom[image_index]:
                            if circle[0] >= self.x_left[image_index]:
                                if circle[0] <= self.x_right[image_index]:
                                    circle_counter += 1
                num_circles_per_image.append(circle_counter)
            image_index += 1
        return num_circles_per_image

    # detektiert Ecken im Bild
    def find_corners(self):
        num_corners_per_image = []
        image_index = 0
        for image in self.images:
            corners_per_image = cv.goodFeaturesToTrack(image, 1000, 0.65, 5)
            if corners_per_image is None:
                num_corners_per_image.append(0)
            else:
                corner_counter = 0
                for corner in corners_per_image:
                    corner = corner[0]
                    if corner[1] <= (self.y_top[image_index] + self.y_bottom[image_index]) / 2:
                        if corner[1] >= self.y_bottom[image_index]:
                            if corner[0] >= self.x_left[image_index]:
                                if corner[0] <= self.x_right[image_index]:
                                    corner_counter += 1
                num_corners_per_image.append(corner_counter)
            image_index += 1
        return num_corners_per_image

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
            self.x_left.append(x_left)
            self.x_right.append(x_right)
            self.y_bottom.append(y_bottom)
            self.y_top.append(y_top)
            width = x_right - x_left
            height = y_top - y_bottom
            current_aspect_ratio = width/height
            aspect_ratios.append(current_aspect_ratio)
        return aspect_ratios

    # detektiert Keypoints anhand Kontrastwechsel
    def find_keypoint(self):
        num_keypoints_per_image = []
        image_index = 0
        for image in self.images:
            sift = cv.SIFT_create()
            kp_per_image = sift.detect(image, None)
            if kp_per_image is None:
                num_keypoints_per_image.append(0)
            else:
                kp_counter = 0
                for kp in kp_per_image:             # konvertiert die Keypoints in x- und y- Koordinaten
                    x = kp.pt[0]
                    y = kp.pt[1]
                    if y <= (self.y_top[image_index] + self.y_bottom[image_index]) / 2:     # überprüft, ob die Keypoints auf der Flasche sind
                        if y >= self.y_bottom[image_index]:
                            if x >= self.x_left[image_index]:
                                if x <= self.x_right[image_index]:
                                    kp_counter += 1
                num_keypoints_per_image.append(kp_counter)
            image_index += 1
        return num_keypoints_per_image

    # detektiert Linien
    def find_lines(self):
        num_lines_per_image = []
        image_index = 0
        for image in self.images:
            edges = cv.Canny(image, 100, 300, apertureSize=3)
            lines_per_image = cv.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
            if lines_per_image is None:
                num_lines_per_image.append(0)
            else:
                line_counter = 0
                for line in lines_per_image:
                    line = line[0]
                    if line[1] <= (self.y_top[image_index] + self.y_bottom[image_index]) / 2:  # überprüft, ob der Startpunkt der Linie auf der Flasche liegt
                        if line[1] >= self.y_bottom[image_index]:
                            line_counter += 1
                    else:
                        if line[3] <= (self.y_top[image_index] + self.y_bottom[
                            image_index]) / 2:  # überprüft, ob der Startpunkt der Linie auf der Flasche liegt
                            if line[3] >= self.y_bottom[image_index]:
                                line_counter += 1
                num_lines_per_image.append(line_counter)
            image_index += 1
        return num_lines_per_image