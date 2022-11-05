import os
from predict import Predict


def main():
    dir_path = os.getcwd()
    test_dataset_path = os.path.join(dir_path, 'small_dataset')  # Caminho para as imagens do banco de dados
    real_dataset_path = os.path.join(dir_path, 'test_dataset')  # Caminho para as imagens reais das cartas para teste
    result_training_path = os.path.join(dir_path, 'training_results')  # Caminho para os modelos treinados
    predict_dict_path = os.path.join(dir_path, 'predicts')  # Caminho para ojson com as informacoes das cartas
    image_real = os.path.join(dir_path, 'test_dataset/2511.png')  # Caminho para as imagens do banco de dados
    image_dataset = os.path.join(dir_path, 'small_dataset/2511.jpg')  # Caminho para as imagens reais das cartas para teste
    model_test = os.path.join(dir_path, 'training_results/ph_teste_v1/50.pth')  # Caminho para os modelos treinados

    predict_card = Predict(real_dataset_path, test_dataset_path, predict_dict_path, model_test)
    dist = predict_card.teste_duas_cartas(image_real, image_dataset, model_test)
    print(dist)


if __name__ == '__main__':
    main()
