from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# LBP Parameters
radius = 1
n_points = 8 * radius
method = 'uniform'

def extract_lbp_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method).astype(np.uint8)

    hist = cv2.calcHist([lbp], [0], None, [n_points + 2], [0, n_points + 2])
    hist = cv2.normalize(hist, hist).flatten()

    return hist



