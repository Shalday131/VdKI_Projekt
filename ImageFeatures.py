# Beschreibung der Funktionen
# Diese Klasse ließt die Features aus den eingelesenen Bilder heraus. Folgende Features werden rausgelesen:
# Anzahl an Kreisen, Anzahl an Ecken, Anzahl an Linien, Anzahl an Orbs, Anzahl an Keypoints, Aspect Ratio
# von dem kleinsten Rechteck, der das Object umgibt und der Maximale Histogrammwert.

# Autoren: Barabanow, Günter, Kauff, Sachweh

import cv2 as cv
import numpy as np

class ImageFeatures:
    def __init__(self):
        self.image = None
        # Eckpunkte für die jeweiligen Rechtecke um die Flasche herum abspeichern
        self.x_left = None
        self.x_right = None
        self.y_bottom = None
        self.y_top = None

    # berechnet die Eckpunkte für das kleinste Rechteck, dass die Flasche umgibt
    def calculate_dimensions(self, image):
        self.image = image
        edges = cv.Canny(self.image, 100, 300)  # hebt die Umrisse des Objekts auf dem Bild hervor
        contours, _ = cv.findContours(edges, cv.RETR_TREE,
                                      cv.CHAIN_APPROX_SIMPLE)  # finde alle Konturen des Objekts im Bild
        cnt = contours[0]
        x_left, y_bottom, x_right, y_top = cv.boundingRect(cnt)
        x_right = 0
        y_top = 0
        for cnt2 in contours:                   # berechnet das kleinste Rechteck, dass das Objekt im Bild umgibt
            x, y, w, h = cv.boundingRect(cnt2)
            if x < x_left:                      # suche kleinstes x
                x_left = x
            if y < y_bottom:                    # suche kleinstes y
                y_bottom = y
            if x + w > x_right:                 # suche größtes x
                x_right = x + w
            if y + h > y_top:                   # suche größtes y
                y_top = y + h
        self.x_left = x_left
        self.x_right = x_right
        self.y_bottom = y_bottom
        self.y_top = y_top

    # detektiert die Kreise, die sich auf der Fasche befinden
    def find_circles(self):
        circles = cv.HoughCircles(self.image, method=cv.HOUGH_GRADIENT_ALT, dp=1.5, minDist=25, param1=140, param2=0.5,
                                  minRadius=1, maxRadius=200)  # hier können die Parameter der Kreisfindung eingestellt werden
        if circles is None:
            return 0
        else:
            circles = circles[0]
            circle_counter = 0
            for circle in circles:
                if circle[1] <= (self.y_top + self.y_bottom) / 2:
                    if circle[1] >= self.y_bottom:
                        if circle[0] >= self.x_left:
                            if circle[0] <= self.x_right:
                                circle_counter += 1
        return circle_counter

    # detektiert Ecken, die sich auf der Flasche befinden
    def find_corners(self):
        corners = cv.goodFeaturesToTrack(self.image, 1000, 0.65, 5)
        if corners is None:
            return 0
        else:
            corner_counter = 0
            for corner in corners:      # überprüft, ob die gefundene Ecke sich auf der Flasche befindet
                corner = corner[0]
                if corner[1] <= (self.y_top + self.y_bottom) / 2:
                    if corner[1] >= self.y_bottom:
                        if corner[0] >= self.x_left:
                            if corner[0] <= self.x_right:
                                corner_counter += 1
        return corner_counter

    # detektiert Linien, die Entweder auf der Flasche anfangen oder auf der Flasche enden
    def find_lines(self):
        edges = cv.Canny(self.image, 100, 300, apertureSize=3)
        lines = cv.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        if lines is None:
            return 0
        else:
            line_counter = 0
            for line in lines:
                line = line[0]
                if line[1] <= (self.y_top + self.y_bottom) / 2:         # überprüft, ob der Startpunkt der Linie auf der
                                                                        # Flasche liegt
                    if line[1] >= self.y_bottom:
                        line_counter += 1
                else:
                    if line[3] <= (self.y_top + self.y_bottom) / 2:     # überprüft, ob der Endpunkt der Linie auf der
                                                                        # Flasche liegt
                        if line[3] >= self.y_bottom:
                            line_counter += 1
        return line_counter

    # findet Orbs, die sich auf der Flasche befinden
    def find_orbs(self):
        orb = cv.ORB_create()                       # initiiere Orbdetektor
        keypoints = orb.detect(self.image, None)         # findet Keypoints mit Orbs
        # compute the descriptors with ORB
        keypoints, des = orb.compute(self.image, keypoints)
        if keypoints is None:
            return 0
        else:
            orb_counter = 0
            for kp in keypoints:    # überprüft, ob die Keypoints auf der Flasche sind
                x = kp.pt[0]        # konvertiert die Keypoints in x- und y- Koordinaten
                y = kp.pt[1]
                if y <= (self.y_top + self.y_bottom) / 2:
                    if y >= self.y_bottom:
                        if x >= self.x_left:
                            if x <= self.x_right:
                                orb_counter += 1
        return orb_counter

    # detektiert Keypoints anhand Kontrastwechsel
    def find_keypoints(self):
            sift = cv.SIFT_create()
            keypoints = sift.detect(self.image, None)
            if keypoints is None:
                return 0
            else:
                kp_counter = 0
                for kp in keypoints:    # überprüft, ob die Keypoints auf der Flasche sind
                    x = kp.pt[0]        # konvertiert die Keypoints in x- und y- Koordinaten
                    y = kp.pt[1]
                    if y <= (self.y_top + self.y_bottom) / 2:
                        if y >= self.y_bottom:
                            if x >= self.x_left:
                                if x <= self.x_right:
                                    kp_counter += 1
            return kp_counter

    # berechnet das Verhältnis zwischen Breite und Höhe des der Flasche umgebenden Rechtecks
    def get_aspect_ratio(self):
        width = self.x_right - self.x_left
        height = self.y_top - self.y_bottom
        aspect_ratio = width / height
        return aspect_ratio

    # findet den maximalen Wert des Histogramms
    def find_max_value_of_histogram(self):
        mask = np.zeros(self.image.shape[:2], np.uint8)                     # kreiere Maske
        mask[self.y_bottom:self.y_top, self.x_left:self.x_right] = 255
        hist_mask = cv.calcHist([self.image], [0], mask, [256], [0, 256])   # berechne Histogramm mit Maske
        hist = [val[0] for val in hist_mask]        # konvertiere Histogramm zu einer Liste
        indices = list(range(0, 256))               # generiere eine Liste mit Indizes
        s = [(x, y) for y, x in sorted(zip(hist, indices), reverse=True)]   # sortiere die Histogrammliste absteigend
        index_of_highest_peak = s[0][0]             # Index des größten Wertes im Histogramm
        max_values_per_image = hist[index_of_highest_peak]
        return max_values_per_image
