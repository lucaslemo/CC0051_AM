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
    return image


def create_positives(image):
    height, width = image.shape[:2]
    positive_images = []
    for i in range(4):
        sig1 = -1 if i < 2 else 1
        sig2 = -1 if (i % 2) == 0 else (i % 2)
        displacement = np.float32([[1, 0, 50 * sig1], [0, 1, 50 * sig2]])
        positive_images.append(cv2.warpAffine(image, displacement, (width, height)))
    center = (height / 2, width / 2)
    for i in range(2):
        sig = -1 if (i % 2) == 0 else (i % 2)
        rotation = cv2.getRotationMatrix2D(center, -10 * sig, 1.0)
        positive_images.append(cv2.warpAffine(image, rotation, (width, height)))
    positive_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return positive_images


def show_image(image, gray=False):
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def main():
    images_path = load_paths()
    for image_path in images_path:
        image = load_image(image_path)
        positive_images = create_positives(image)
        for image in positive_images:
            if len(image.shape) == 2:
                show_image(image, gray=True)
            else:
                show_image(image)
        break


if __name__ == '__main__':
    main()
