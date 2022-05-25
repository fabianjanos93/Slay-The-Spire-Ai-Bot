import cv2
import numpy as np


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)

# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 250, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def normalize_name(image):
    return invert(filling(thresholding(get_grayscale(image)), (50, 10)))


def normalize_cost(image):
    img_1 = thresholding(get_grayscale(image))
    # img_2 = filling(img_1, (3, 30))
    # img_3 = filling(img_2, (40, 10))
    return invert(img_1)


def filling(image, position):
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(image, mask, position, 0)
    return image

def invert(image):
    return cv2.bitwise_not(image)

# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def invert(image):
    return cv2.bitwise_not(image)
