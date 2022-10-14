import os
import random
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
    train = Train(images_path)
    tarin.init()
    train.start()


if __name__ == '__main__':
    main()
