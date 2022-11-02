import os
from training import Train
from predict import Predict


def main():
    dir_path = os.getcwd()
    data_training_path = os.path.join(dir_path, 'train_test_dataset')
    # data_test_path = os.path.join(dir_path, 'test_dataset') # Cartas do banco modificadas
    data_test_path = os.path.join(dir_path, 'real_dataset') # Imagens reais das cartas
    result_training_path = os.path.join(dir_path, 'training_results')

    # Train
    # train = Train(data_training_path, number_epochs=1)
    # train.start()

    # Predict
    predict_card = Predict(data_test_path)
    predict_card.test_training()


if __name__ == '__main__':
    main()
