import cv2 as cv

import image_organisation as imgorga
import image_preprocessing as imgprep

# Bilder holen
test_imgorga = imgorga.image_organisation()
images = test_imgorga.get_images()

#Bilder vorbereiten
test_imgprep = imgprep.image_preprocessing(images)

test_imgprep.resize()
test_imgprep.grayscale()
test_imgprep.blur()
#test_imgprep.thresholding()

modified_images = test_imgprep.get_modified_images()
# cv.imshow("test", modified_images[50])
# cv.waitKey(0)
# cv.destroyAllWindows()

test_imgprep.HoughCircles()