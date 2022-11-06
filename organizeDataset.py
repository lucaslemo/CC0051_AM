import os
import shutil


def main():
    # Arrumando as imagens no dataset
    dir = os.getcwd()
    path = os.path.join(dir, 'small_dataset')

    for card in os.listdir(path):
        file_path = os.path.join(path, card)
        folder_name = card.strip('.jpg')
        destino = os.path.join(path, folder_name)
        try:
            os.makedirs(destino)
            shutil.move(file_path, destino)
        except:
            exit()


if __name__ == '__main__':
    main()
