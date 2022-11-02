import os
import random
import torch
import torch.nn as torch_nn
import torchvision.models as models
import torch.nn.functional as nn_functional
from torch import optim, save as save
from convertImage import ConvertImage as imageClass


class Predict:
    def __init__(self, cards_path):
        self.net = SiameseNetwork()
        self.data = self.__load_data_paths(cards_path)
        self.number_cards = len(self.data)

    def __load_data_paths(self, cards_path):
        images_path = []
        for card in os.listdir(cards_path):
            item = {
                'path': os.path.join(cards_path, card),
                'card_id': int(card.strip('.jpg')),
                'status_use': 0
            }
            images_path.append(item)
        return images_path

    def __getSimilarRank(self, image, image2):
        output1, output2 = self.net(image, image2)
        euclidean_distance = nn_functional.pairwise_distance(output1, output2)
        return euclidean_distance, (output1, output2)

    def test_training(self):
        for card in range(self.number_cards):
            img_card = imageClass(self.data[card]['path']).get_anchor()
            for i in range(self.number_cards):
                data = imageClass(self.data[i]['path']).get_anchor()
                print(str(card) + ' -> ' + str(i) + ':', self.__getSimilarRank(img_card, data)[0])
            print()
        

    def start(self):
        img_card = imageClass(self.data[0]['path'])
        img_test = imageClass(self.data[1]['path'])
        print(self.__getSimilarRank(img_card, img_card)[0])
        print(self.__getSimilarRank(img_card, img_test)[0])


class SiameseNetwork(torch_nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.resnet.load_state_dict(torch.load('./training_results/10Cartas1epocasRESNET101.pth'), strict=False)
        self.resnet.eval()

    def forward_once(self, input1):
        output = self.resnet(input1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
