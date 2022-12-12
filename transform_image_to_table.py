import cv2 as cv
import pandas as pd

from ImageOrganisation import ImageOrganisation
from ImagePreprocessing import ImagePreprocessing
from ImageFeatures import ImageFeatures
from TestFeatures import TestFeatures

# Bilder holen
imgorga = ImageOrganisation()
imgprep = ImagePreprocessing()
imgfeatures = ImageFeatures()
# images = imgorga.get_images()

image_counter = 0
for path in imgorga.image_paths:
    image = cv.imread(path)

    # Bilder vorbereiten
    modified_image = imgprep.preprocessing(image)

    # Label von dem Bild rauslesen
    label = imgorga.get_label(path)

    # Features rauslesen
    imgfeatures.calculate_dimensions(modified_image)
    num_circles = imgfeatures.find_circles()
    num_corners = imgfeatures.find_corners()
    num_lines = imgfeatures.find_lines()
    num_orbs = imgfeatures.find_orbs()
    num_keypoints = imgfeatures.find_keypoints()
    aspect_ratio = imgfeatures.get_aspect_ratio()
    max_value_of_histogram = imgfeatures.find_max_value_of_histogram()

    # sammle gefundene Features in einem Array
    imgorga.collect_features([num_circles, num_corners, num_lines, num_orbs, num_keypoints, aspect_ratio,
                                  max_value_of_histogram, label])

    image_counter += 1
    if image_counter == 1:
        print("Es wurde", image_counter, "Bild eingelesen.")
    else:
        print("Es wurden", image_counter, "Bilder eingelesen.")

column_names = ["Anzahl Kreise", "Anzahl Ecken", "Anzahl Linien", "Anzahl Orbs", "Anzahl Keypoints", "Aspect Ratio",
                "Maximaler Histogrammwert", "Label"]
df = imgorga.get_dataframe(column_names)
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
