import cv2 as cv
from matplotlib import pyplot as plt

from ImageOrganisation import ImageOrganisation
from ImagePreprocessing import ImagePreprocessing
from TestFeatures import TestFeatures

# Bilder holen
test_imgorga = ImageOrganisation()
images = test_imgorga.get_images()

# Bilder vorbereiten
test_imgprep = ImagePreprocessing(images)
test_imgprep.resize()
test_imgprep.grayscale()
test_imgprep.blur()
# test_imgprep.thresholding()
modified_images = test_imgprep.get_modified_images()
test_imgprep.find_contours()

# Feature Tests:
test_features = TestFeatures(modified_images)
# test_features.find_circles_test()
# test_features.find_corners_test()
# test_features.find_edges_test()
# test_features.SIFT_test()
# test_features.find_contours_test()
