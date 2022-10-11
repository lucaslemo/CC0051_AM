import os
import numpy as np
import matplotlib.pyplot as plt
import random
from Image import Image


def load_paths():
    dir_path = os.getcwd()
    data_path = os.path.join(dir_path, 'dataset')
    images_path = []
    for card in os.listdir(data_path):
        images_path.append(os.path.join(data_path, card))
    return images_path


def show_image(image, gray=True):
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def main():
    images_path = load_paths()
    value_image = random.randint(0, 12364)
    image = Image(images_path[value_image])
    neg_value = random.randint(0, 12364)
    while neg_value == value_image:
        neg_value = random.randint(0, 12364)
    image_neg = Image(images_path[neg_value])
    img_pos = np.hstack((image.get_anchor(), image.get_positive()))
    img_neg = np.hstack((image_neg.get_anchor(), image_neg.get_positive()))
    show_image(np.hstack((img_pos, img_neg)))


if __name__ == '__main__':
    main()
