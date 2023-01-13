# Beschreibung der Funktionen
# Mit Hilfe dieser Klasse werden alle Bilder auf die selbe Größe gebracht, in Gratöne konvertiert und ein
# Unschärfefilter drübergelegt.
#
# Autoren: Barabanow, Günter, Kauff, Sachweh

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

class ImagePreprocessing:

    def __init__(self):
        # Eckpunkte für die jeweiligen Rechtecke um die Flasche herum abspeichern
        self.x_left = []
        self.x_right = []
        self.y_bottom = []
        self.y_top = []

    def preprocessing(self, image):
        resize = cv.resize(image, (700, 700))
        gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        return blur
