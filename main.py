import os
import random
from Image import Image
from training import Train


def load_paths():
    dir_path = os.getcwd()
    data_path = os.path.join(dir_path, 'dataset')
    images_path = []
    for card in os.listdir(data_path):
        images_path.append(os.path.join(data_path, card))
    return images_path


def main():
    images_path = load_paths()
    value_image = random.randint(0, 12364)
    neg_value = random.randint(0, 12364)
    image = Image(images_path[value_image])
    while neg_value == value_image:
        neg_value = random.randint(0, 12364)
    image_neg = Image(images_path[neg_value])
    train = Train()
    train.start(image.get_anchor(), image.get_positive(), image_neg.get_anchor())


if __name__ == '__main__':
    main()
