import numpy as np
import random
import cv2


class Image:

    def __init__(self, image_path):
        origin = cv2.imread(image_path)
        image1 = origin[110:430, 50:370, :]
        image2 = self.select_positive(image1, random.randint(1, 4))
        self.image_ach = self.__rgb2gray(image1)
        self.image_pos = self.__rgb2gray(image2)

    def select_positive(self, image, value):
        if value == 1:
            return self.__increase_bright(image)
        elif value == 2:
            return self.__decrease_bright(image)
        elif value == 3:
            return self.__changing_contrast(image)
        elif value == 4:
            return self.__changing_bright(image)

    def __increase_bright(self, image):
        value_plus = random.randint(50, 80)
        m = np.ones(image.shape, dtype="uint8") * value_plus
        added = cv2.add(image, m)
        return added

    def __decrease_bright(self, image):
        value_minus = random.randint(50, 80)
        m = np.ones(image.shape, dtype="uint8") * value_minus
        added = cv2.subtract(image, m)
        return added

    def __changing_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        i = random.randint(1, 10)
        clp = -10.0 * i - 4
        tgs = (i, i)
        clahe = cv2.createCLAHE(clipLimit=clp, tileGridSize=tgs)
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        result = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return result

    def __changing_bright(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        i = random.randint(1, 10)
        lim = 255 - i
        v[v > lim] = 255
        v[v <= lim] += i
        final_hsv = cv2.merge((h, s, v))
        result = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return result

    def __rgb2gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def get_anchor(self):
        return self.image_ach

    def get_positive(self):
        return self.image_pos
