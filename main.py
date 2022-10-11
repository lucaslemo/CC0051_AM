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
        clp = -10.0*i - 4
        tgs = (i, i)
        clahe = cv2.createCLAHE(clipLimit=clp, tileGridSize=tgs)
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        result = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(img_crp, cv2.COLOR_BGR2GRAY)
        a = np.hstack((gray, result))
        img_pos.append(a)
    # Brilho
    hsv = cv2.cvtColor(img_crp, cv2.COLOR_BGR2HSV)
    for i in range(10, 100, 30):
        h, s, v = cv2.split(hsv)
        lim = 255 - i
        v[v > lim] = 255
        v[v <= lim] += i
        final_hsv = cv2.merge((h, s, v))
        enhanced_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        result = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(img_crp, cv2.COLOR_BGR2GRAY)
        a = np.hstack((gray, result))
        img_pos.append(a)
    return img_pos


def changing_contrast(image_path):
    image = imread(image_path)
    img_crp = image[110:430, 50:370, :]
    new_image = increase_bright(img_crp, 50)
    lab = cv2.cvtColor(new_image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    i = random.randint(1, 10)
    clp = -10.0 * i - 4
    tgs = (i, i)
    clahe = cv2.createCLAHE(clipLimit=clp, tileGridSize=tgs)
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    result = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    return result


def changing_bright(image_path):
    image = imread(image_path)
    img_crp = image[110:430, 50:370, :]
    new_image = decrease_bright(img_crp, 50)
    hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    i = random.randint(1, 10)
    lim = 255 - i
    v[v > lim] = 255
    v[v <= lim] += i
    final_hsv = cv2.merge((h, s, v))
    enhanced_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    result = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    return result


def show_image(image, gray=True):
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def increase_bright(image, value):
    m = np.ones(image.shape, dtype="uint8") * value
    added = cv2.add(image, m)
    return added


def decrease_bright(image, value):
    m = np.ones(image.shape, dtype="uint8") * value
    added = cv2.subtract(image, m)
    return added


def main():
    images_path = load_paths()
    for i in range(10):
        cont = 0
        value_image = random.randint(0, 12000)
        for image_path in images_path:
            if cont == value_image:
                value_plus = random.randint(20, 80)
                value_minus = random.randint(20, 80)
                img_ach = load_image(image_path)
                #img_pos = create_positives(image_path)
                img_plus = increase_bright(img_ach, value_plus)
                img_minus = decrease_bright(img_ach, value_minus)
                img_contrast = changing_contrast(image_path)
                img_bright = changing_bright(image_path)
                img = np.hstack((img_plus, img_contrast))
                img = np.hstack((img, img_ach))
                img = np.hstack((img, img_bright))
                show_image(np.hstack((img, img_minus)))
                break
            cont += 1


if __name__ == '__main__':
    main()
