import cv2 as cv
import pandas as pd
from matplotlib import pyplot as plt

from ImageOrganisation import ImageOrganisation
from ImagePreprocessing import ImagePreprocessing
from TestFeatures import TestFeatures

# Bilder holen
test_imgorga = ImageOrganisation()
images = test_imgorga.get_images()

# Labels zu den Bildern holen
labels = test_imgorga.get_labels()


# Bilder vorbereiten
test_imgprep = ImagePreprocessing(images)
test_imgprep.resize()
test_imgprep.grayscale()
test_imgprep.blur()
# test_imgprep.thresholding()
modified_images = test_imgprep.get_modified_images()

# Features rauslesen
aspect_ratio = test_imgprep.find_contours() # hier werden die Eckpunkte der Rechtecke berechnet
num_circles = test_imgprep.find_circles()


# Feature Tests:
test_features = TestFeatures(modified_images)
# test_features.find_circles_test()
# test_features.find_corners_test()
test_features.find_edges_test()
# test_features.SIFT_test()
test_features.find_contours_test()

# Datafreame erzeugen
df = pd.DataFrame({"Anzahl Kreise": num_circles, "Aspect Ratio": aspect_ratio, "Labels": labels})
df.to_excel("Features_Test.xlsx")
