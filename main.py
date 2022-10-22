import os
from training import Train
from predict import Predict


def main():
    dir_path = os.getcwd()
    data_training_path = os.path.join(dir_path, 'train_test_dataset')
    result_training_path = os.path.join(dir_path, 'training_results')

    # Train

    train = Train(data_training_path, number_epochs=1)
    train.start()


    # Predict
    '''
    predict_card = Predict(data_training_path)
    s = predict_card.test_training()
    for i in range(len(s)):
        print('Carta {}: '.format(i), end='')
        print(s[i])
    '''

if __name__ == '__main__':
    main()
