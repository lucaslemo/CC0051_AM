import gc
import os
import time
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import cv2
from matplotlib.pyplot import imread


def load_paths():
    dir_path = os.getcwd()
    data_path = os.path.join(dir_path, 'dataset')
    images_path = []
    for card in os.listdir(data_path):
        images_path.append(os.path.join(data_path, card))
    return images_path


def load_image(image_path):
    image = imread(image_path)
    image_cropped = image[110:430, 50:370, :]
    image_result = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
    return image_result


def create_positives(image_path):
    image = imread(image_path)
    img_crp = image[110:430, 50:370, :]
    img_pos = []
    # Contraste
    lab = cv2.cvtColor(img_crp, cv2.COLOR_BGR2LAB)
    for i in range(1, 10, 3):
        l_channel, a, b = cv2.split(lab)
        clp = 1.0*i - 4
        tgs = (i, i)
        clahe = cv2.createCLAHE(clipLimit=clp, tileGridSize=tgs)
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        result = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        img_pos.append(result)
    # Brilho
    hsv = cv2.cvtColor(img_crp, cv2.COLOR_BGR2HSV)
    for i in range(10, 50, 10):
        h, s, v = cv2.split(hsv)
        lim = 255 - i
        v[v > lim] = 255
        v[v <= lim] += i
        final_hsv = cv2.merge((h, s, v))
        enhanced_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        result = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        img_pos.append(result)
    return img_pos


def show_image(image, gray=True):
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def main():
    images_path = load_paths()
    for image_path in images_path:
        img_ach = load_image(image_path)
        img_pos = create_positives(image_path)
        show_image(img_ach)
        for image in img_pos:
            show_image(image)
        break


if __name__ == '__main__':
    main()
