import os
from training import Train
from predict import Predict


def main():
    # Caminhos para os arquivos
    dir_path = os.getcwd() # Diretorio raiz
    train_dataset_path = os.path.join(dir_path, 'small_dataset') # Caminho para as imagens do banco de dados
    test_dataset_path = os.path.join(dir_path, 'dataset_test') # Caminho para as imagens do banco de dados para teste
    real_dataset_path = os.path.join(dir_path, 'dataset_real') # Caminho para as imagens reais das cartas

    # Indica se o algoritmo vai treinar ou testar
    train = False

    # Variaveis de Treino
    epochs = 300
    batch_size = 24
    margin = 2.0
    learning_rate = 0.0005

    if train == True:
        # Train
        train = Train(
            training_path=train_dataset_path, 
            number_epochs=epochs, 
            batch_size=batch_size,
            margin=margin,
            learning_rate=learning_rate,
            save_imgs=False
            )
        train.start()
    else:
        models_path = os.path.join(dir_path, 'training_results')
        models_epoch = os.path.join(models_path, 'v3')
        model_list_epoch = {}  # Dicionario dos caminhos para os modelos treinados
        for model_file in os.listdir(models_epoch):
            item = {
                'path': os.path.join(models_epoch, model_file),
                'name': model_file.strip('.pth')
            }
            model_list_epoch[int(model_file.strip('.pth'))] = item

        # Arquivo csv
        file_csv = open('v4new.csv', mode='w')

        # Guardando no csv os valores das epocas trinadas
        for i in range(0, epochs + 1, 10):
            if i == 0:
                print('modelos', file=file_csv, end='; ')
            elif i < epochs:
                print(i, file=file_csv, end='; ')
            else:
                print(i, file=file_csv)

        # Predict
        mean_neg = []
        for i in range(0, epochs + 1, 10):
            if i == 0:
                print('positive', file=file_csv, end='; ')
            else:
                predict_card = Predict(real_dataset_path, test_dataset_path, model_list_epoch[i]['path'])
                mean1, mean2 = predict_card.start()
                mean_neg.append(mean2)
                if i < epochs:
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
