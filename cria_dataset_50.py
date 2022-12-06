import os
import shutil


def main():
    # Variaveis de diretorio (paths)
    dir = os.getcwd()
    dataset = os.path.join(dir, 'dataset')
    main_dataset = os.path.join(dir, 'main_dataset')

    # Le as cartas do arquivo txt
    with open('./Cartas.txt') as cartas:
        lines = [line.rstrip('\n') for line in cartas]

    # Para cada carta pegamos o caminho dela no dataset
    # Criamos os diretorios e copiamos as cartas
    try:
        os.makedirs(main_dataset)
    except:
        exit()
    for line in lines:
        id = line.split(' - ')[0]
        item = id + '.jpg'
        dir_card = os.path.join(main_dataset, id)
        card_path = os.path.join(dataset, item)
        try:
            os.makedirs(dir_card)
            shutil.copy(card_path, dir_card)
        except:
            exit()


if __name__ == '__main__':
    main()
