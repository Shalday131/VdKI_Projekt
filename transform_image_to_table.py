import pandas as pd

from ImageOrganisation import ImageOrganisation
from ImagePreprocessing import ImagePreprocessing
from TestFeatures import TestFeatures

# Bilder holen
imgorga = ImageOrganisation()
images = imgorga.get_images()

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
df.to_excel("Features.xlsx")

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
