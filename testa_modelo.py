import os
from predict import Predict


def main():
    dir_path = os.getcwd()
    dataset = os.path.join(dir_path, 'small_dataset')  # Caminho para as imagens do banco de dados
    data_test_path = os.path.join(dir_path, 'dme')  # Caminho para as imagens reais das cartas para teste
    result_training_path = os.path.join(dir_path, 'training_results')  # Caminho para os modelos treinados
    predict_dict_path = os.path.join(dir_path, 'predicts')  # Caminho para ojson com as informacoes das cartas
    model_list = []  # Lista dos caminhos para os modelos treinados
    for model_file in os.listdir(result_training_path):
        item = {
            'path': os.path.join(result_training_path, model_file),
            'name': model_file.strip('.pth')
        }
        model_list.append(item)

    file_csv = open('meu_csv.csv', mode='w')

    # Predict
    for i in range(0, 301, 10):
        if i == 0:
            print('modelos', file=file_csv, end='; ')
        elif i < 300:
            print(i, file=file_csv, end='; ')
        else:
            print(i, file=file_csv, end='')

    for count in range(0, 31):
        if count == 0:
            print('1m5e-2lr', file=file_csv, end='; ')
        else:
            predict_card = Predict(data_test_path, dataset, predict_dict_path, model_list[count]['path'])
            media = predict_card.model_test()
            if count < 30:
                print(media, file=file_csv, end='; ')
            else:
                print(media, file=file_csv, end='')

    file_csv.close()


if __name__ == '__main__':
    main()
