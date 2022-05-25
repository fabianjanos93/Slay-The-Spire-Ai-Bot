from datetime import time
from random import randint

import cv2
from pytesseract import pytesseract

from pictureNormalize import normalize_name, normalize_cost, get_grayscale

counter = 0


class Card:
    cost = ""
    name = ""
    text = ""

    def __init__(self, image):
        h, w, other = image.shape
        imgText = image[4 * h // 7:h, 0:w]
        imgName = image[20:h // 6, 40:w]
        imgCost = image[0:h // 7, 0:w // 5]

        # psm settings : https://stackoverflow.com/questions/44619077/pytesseract-ocr-multiple-config-options
        single_character = '--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
        number_character = r'--oem 3 --psm 7 '
        text = get_grayscale(imgText)
        self.text = pytesseract.image_to_string(text)

        name = normalize_name(imgName)
        self.name = pytesseract.image_to_string(name)

        cost = normalize_cost(imgCost)
        self.cost = pytesseract.image_to_string(cost, config=single_character)
        s = str(randint(0, 100))
        print(s)
        path = "../resources/"
        cv2.imwrite(path + "text-normalized" + s + ".png", text)
        cv2.imwrite(path + "name-normalized" + s + ".png", name)
        cv2.imwrite(path + "cost-normalized" + s + ".png", cost)

        print("cost: ", self.cost, " name: ", self.name, " text: ", self.text)
        print("------")
