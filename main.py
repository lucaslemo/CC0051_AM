import os
from training import Train
from predict import Predict


def main():
    dir_path = os.getcwd()
    data_training_path = os.path.join(dir_path, 'train_test_dataset') # Caminho para as imagens do banco para treino
    data_test_path = os.path.join(dir_path, 'real_dataset') # Caminho para as imagens reais das cartas para teste
    result_training_path = os.path.join(dir_path, 'training_results') # Caminho para os modelos treinados
    model_list = [] # Lista dos caminhos para os modelos treinados
    for model_file in os.listdir(result_training_path):
            item = {
                'path': os.path.join(result_training_path, model_file),
                'name': model_file.strip('.pth')
            }
            model_list.append(item)

    # Train
    # train = Train(data_training_path, number_epochs=1)
    # train.start()

    # Predict
    for model in model_list:
        print(model['name'])
        predict_card = Predict(data_test_path, model['path'])
        predict_card.test_training()


if __name__ == '__main__':
    main()
