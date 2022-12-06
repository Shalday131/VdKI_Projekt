import cv2 as cv

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

# Feature Tests:
test_features = TestFeatures(modified_images)
test_features.find_circles_test()
test_features.find_corners_test()

# cv.imshow("test", modified_images[50])
# cv.waitKey(0)
# cv.destroyAllWindows()

# test_imgprep.find_corners()