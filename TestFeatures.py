import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy

class TestFeatures:

    def __init__(self, images):
        self.images = images
        self.current_index = 63
        self.edge_image = None
        self.x_left = None
        self.x_right = None
        self.y_bottom = None
        self.y_top = None

    def find_circles_test(self):
        clone = copy.deepcopy(self.images[self.current_index])
        circles = cv.HoughCircles(clone, cv.HOUGH_GRADIENT_ALT, dp=1.5, minDist=25, param1=140, param2=0.5, minRadius=1, maxRadius=200)
        circles = np.uint16(np.around(circles))
        circles = circles[0]
        print("Kreise: ", circles)
        print("Anzahl Kreise: ", len(circles)) # für jeden Kreis werden 3 Zahlen gespeichert --> durch 3 teilen
        # Bild darstellen
        for i in circles:
            # draw the outer circle
            cv.circle(clone, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv.circle(clone, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv.imshow('detected circles', clone)
        cv.waitKey(0)
        cv.destroyAllWindows()


    def find_corners_test(self):
        clone = copy.deepcopy(self.images[self.current_index])
        corners = cv.goodFeaturesToTrack(clone, 1000, 0.65, 5)
        corners = np.int0(corners)
        print(corners)
        # Bild darstellen
        for i in corners:
            x, y = i.ravel()
            cv.circle(clone, (x, y), 3, 255, -1)
        cv.imshow("Corners", clone)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_edges_test(self):
        clone = copy.deepcopy(self.images[self.current_index])
        edges = cv.Canny(clone, 100, 300)
        self.edge_image = edges
        # Bild darstellen
        plt.subplot(121), plt.imshow(clone, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def SIFT_test(self):
        clone = copy.deepcopy(self.images[self.current_index])
        sift = cv.SIFT_create()
        print(sift)
        kp = sift.detect(clone, None)
        print(kp)
        x = []
        y = []
        for keypoint in kp:             # Konvertiert die Keypoints in x- und y- Koordinaten
            x.append(keypoint.pt[0])
            y.append(keypoint.pt[1])
        print("x: ", x)
        print("y: ", y)
        # Zeichnet die Features in das Bild
        clone = cv.drawKeypoints(clone, kp, self.images[self.current_index], flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow('SIFT', clone)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_contours_test(self):
        img = self.edge_image

        # find the contours
        contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # take the first contour
        cnt = contours[0]

        x_left, y_bottom, x_right, y_top = cv.boundingRect(cnt)
        x_right = 0
        y_top = 0
        # compute the bounding rectangle of the contour
        for cnt2 in contours:
            x, y, w, h = cv.boundingRect(cnt2)
            if x < x_left:    # suche kleinstes x
                x_left = x
            if y < y_bottom:    # suche kleinstes y
                y_bottom = y
            if x+w > x_right:            # suche größtes x
                x_right = x+w
            if y+h > y_top:            # suche größtes y
                y_top = y+h
        self.x_left = x_left
        self.y_bottom = y_bottom
        self.x_right = x_right
        self.y_top = y_top
        print(y_top, y_bottom)
        print((y_top+y_bottom)/2)
        # draw contour
        img = cv.drawContours(img, [cnt], 0, (0, 255, 255), 2)

        # draw the bounding rectangle
        img = cv.rectangle(img, (x_left, y_bottom), (x_right, y_top), (255, 50, 255), 2)

        # display the image with bounding rectangle drawn on it
        cv.imshow("Bounding Rectangle", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def find_lines_test(self):
        clone = copy.deepcopy(self.images[self.current_index])
        edges = cv.Canny(clone, 100, 300, apertureSize=3)
        lines = cv.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        print(lines)
        print(len(lines))
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(clone, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv.imshow("lines", clone)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def create_histogram_test(self):
        clone = copy.deepcopy(self.images[self.current_index])
        # create a mask
        mask = np.zeros(clone.shape[:2], np.uint8)
        mask[self.y_bottom:self.y_top, self.x_left:self.x_right] = 255
        masked_img = cv.bitwise_and(clone, clone, mask=mask)
        # Calculate histogram with mask and without mask
        # Check third argument for mask
        hist_full = cv.calcHist([clone], [0], None, [256], [0, 256])
        hist_mask = cv.calcHist([clone], [0], mask, [256], [0, 256])

        # Convert histogram to simple list
        hist = [val[0] for val in hist_mask]
        # Generate a list of indices
        indices = list(range(0, 256))
        # Descending sort-by-key with histogram value as key
        s = [(x, y) for y, x in sorted(zip(hist, indices), reverse=True)]
        # Index of highest peak in histogram
        index_of_highest_peak = s[0][0]
        print("index of highest peak: ", index_of_highest_peak)
        print(hist[index_of_highest_peak])

        plt.subplot(221), plt.imshow(clone, 'gray')
        plt.subplot(222), plt.imshow(mask, 'gray')
        plt.subplot(223), plt.imshow(masked_img, 'gray')
        plt.subplot(224), plt.plot(hist_mask)
        plt.xlim([0, 256])
        plt.show()

    def find_orbs_test(self):
        clone = copy.deepcopy(self.images[self.current_index])
        # Initiate ORB detector
        orb = cv.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(clone, None)
        # compute the descriptors with ORB
        kp, des = orb.compute(clone, kp)
        x = []
        y = []
        for keypoint in kp:  # Konvertiert die Keypoints in x- und y- Koordinaten
            x.append(keypoint.pt[0])
            y.append(keypoint.pt[1])
        print("x: ", x)
        print("y: ", y)
        # draw only keypoints location,not size and orientation
        img2 = cv.drawKeypoints(clone, kp, None, color=(0, 255, 0), flags=0)
        plt.imshow(img2), plt.show()