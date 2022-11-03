import os
from training import Train
from predict import Predict


def main():
    dir_path = os.getcwd()
    dataset = os.path.join(dir_path, 'small_dataset') # Caminho para as imagens do banco de dados
    data_test_path = os.path.join(dir_path, 'dme') # Caminho para as imagens reais das cartas para teste
    result_training_path = os.path.join(dir_path, 'training_results') # Caminho para os modelos treinados
    predict_dict_path = os.path.join(dir_path, 'predicts') # Caminho para ojson com as informacoes das cartas
    model_list = [] # Lista dos caminhos para os modelos treinados
    for model_file in os.listdir(result_training_path):
            item = {
                'path': os.path.join(result_training_path, model_file),
                'name': model_file.strip('.pth')
            }
            model_list.append(item)

    # Train
    train = Train(dataset, number_epochs=300)
    train.start()

    # Predict
    # predict_card = Predict(data_test_path, dataset, predict_dict_path)
    # predict_card.test_training()


if __name__ == '__main__':
    main()
