import cv2 as cv
import pandas as pd

from ImageOrganisation import ImageOrganisation
from ImagePreprocessing import ImagePreprocessing
from TestFeatures import TestFeatures

# Bilder holen
imgorga = ImageOrganisation()
# images = imgorga.get_images()

for path in imgorga.image_paths():
    image = cv.imread(path)

    # Bilder vorbereiten
    imgprep = ImagePreprocessing(image)
    imgprep.preprocessing()                 # hier sollen die Bilder resized, in Graubilder umgewandelt und verunschärt werden

    # Label von dem Bild rauslesen
    label = imgorga.get_labels()

    # Features rauslesen
    imgfeatures = ImageFeatures(image)              # neue Klasse die nur die Features beinhaltet, im Kostruktor sollen die x- und y-Werte von dem umgebenden Rechteck berechnet werden
    num_circles = imgfeatures.find_circles()
    num_corners = imgfeatures.find_corners()
    num_lines = imgfeatures.find_lines()
    num_bubbles = imgfeatures.find_bubbles()
    num_keypoints = imgfeatures.find_keypoints()
    aspect_ratio = imgfeatures.get_aspect_ratio()

    imgfeatures.collect_features([num_circles, num_corners, num_lines, num_bubbles, num_keypoints, aspect_ratio])   # hier wird ein Array erstellt, der die gefundenen Features sammelt

'''
# Labels zu den Bildern holen
labels = imgorga.get_labels()


# Bilder vorbereiten
imgprep = ImagePreprocessing(images)
imgprep.resize()
imgprep.grayscale()
imgprep.blur()
# test_imgprep.thresholding()
modified_images = imgprep.get_modified_images()

# Features rauslesen
aspect_ratio = imgprep.find_contours() # hier werden  unter anderem die Eckpunkte der Rechtecke berechnet --> diese Funktion muss immer vor den anderen aufgerufen werden
num_circles = imgprep.find_circles()
num_corners = imgprep.find_corners()
num_keypoints = imgprep.find_keypoint()
num_lines = imgprep.find_lines()
num_orbs = imgprep.find_orbs()
max_values_of_histogram = imgprep.find_max_value_of_histogram()

# Datafreame mit Features erzeugen
df = pd.DataFrame({"Anzahl Kreise": num_circles, "Aspect Ratio": aspect_ratio, "Anzahl Ecken": num_corners,
                   "Anzahl Keypoints": num_keypoints, "Anzahl Linien": num_lines, "Anzahl Orbs": num_orbs,
                   "maximaler Histogrammwert": max_values_of_histogram, "Labels": labels})
df.to_excel("Features_besser.xlsx")

# Feature Tests:
# test_features = TestFeatures(modified_images)
# test_features.find_circles_test()
# test_features.find_corners_test()
# test_features.find_edges_test()
# test_features.SIFT_test()
# test_features.find_contours_test()
# test_features.find_lines_test()
# test_features.create_histogram_test()
# test_features.find_orbs_test()
'''