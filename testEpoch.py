import os
from predict import Predict


def main():
    dir_path = os.getcwd()
    test_dataset_path = os.path.join(dir_path, 'new_dataset')  # Caminho para as imagens do banco de dados
    real_dataset_path = os.path.join(dir_path, 'new_test_dataset')  # Caminho para as imagens reais das cartas para teste
    result_training_path = os.path.join(dir_path, 'training_results')  # Caminho para os modelos treinados
    predict_dict_path = os.path.join(dir_path, 'predicts')  # Caminho para ojson com as informacoes das cartas
    models_epoch = os.path.join(result_training_path, 'ph_teste_v2')
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
    file_csv = open('ph_teste_epoch_v1.csv', mode='w')
    predict_card = Predict(real_dataset_path, test_dataset_path, predict_dict_path, model_list_epoch[250]['path'])
    result = predict_card.calcula_todas_distancias()

    count = 0
    for card in os.listdir(test_dataset_path):
        item = {
            'path': os.path.join(test_dataset_path, card),
            'name': card.strip('.pth')
        }
        if count < 20:
            print(item['name'], file=file_csv, end='; ')
        else:
            print(item['name'], file=file_csv)
        count = count + 1
    for linha in result:
        enum = 0
        for coluna in linha:
            if enum < 20:
                print(coluna, file=file_csv, end='; ')
            else:
                print(coluna, file=file_csv)
            enum = enum + 1

    file_csv.close()


if __name__ == '__main__':
    main()
