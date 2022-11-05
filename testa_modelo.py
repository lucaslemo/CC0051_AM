import os
from predict import Predict


def main():
    dir_path = os.getcwd()
    test_dataset_path = os.path.join(dir_path, 'small_dataset')  # Caminho para as imagens do banco de dados
    real_dataset_path = os.path.join(dir_path, 'test_dataset')  # Caminho para as imagens reais das cartas para teste
    result_training_path = os.path.join(dir_path, 'training_results')  # Caminho para os modelos treinados
    predict_dict_path = os.path.join(dir_path, 'predicts')  # Caminho para ojson com as informacoes das cartas
    models_epoch = os.path.join(result_training_path, 'ph_teste_v1')
    model_list = []  # Lista dos caminhos para os modelos treinados
    for model_file in os.listdir(result_training_path):
        item = {
            'path': os.path.join(result_training_path, model_file),
            'name': model_file.strip('.pth')
        }
        model_list.append(item)

    model_list_epoch = {}  # Dicionario dos caminhos para os modelos treinados
    for model_file in os.listdir(models_epoch):
        item = {
            'path': os.path.join(models_epoch, model_file),
            'name': model_file.strip('.pth')
        }
        model_list_epoch[int(model_file.strip('.pth'))] = item

    # Predict
    file_csv = open('ph_teste_v1.csv', mode='w')
    mean_neg = []
    for i in range(0, 301, 10):
        if i == 0:
            print('positive', file=file_csv, end='; ')
        else:
            predict_card = Predict(real_dataset_path, test_dataset_path, predict_dict_path, model_list_epoch[i]['path'])
            mean1, mean2 = predict_card.start()
            mean_neg.append(mean2)
            if i < 300:
                print(mean1, file=file_csv, end='; ')
            else:
                print(mean1, file=file_csv)
    for i in range(len(mean_neg)):
        if i == 0:
            print('negative', file=file_csv, end='; ')
        elif i < len(mean_neg):
            print(mean_neg[i], file=file_csv, end='; ')
        else:
            print(mean_neg[i], file=file_csv)

    file_csv.close()


if __name__ == '__main__':
    main()
