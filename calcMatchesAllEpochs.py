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
    for i in range(10, 301, 10):
        predict_card = Predict(real_dataset_path, test_dataset_path, predict_dict_path, model_list_epoch[i]['path'])
        qtd = predict_card.testa_acertos()
        print('Epoch {}: {}'.format(i, qtd))


if __name__ == '__main__':
    main()
