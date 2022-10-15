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
        list_ret = []
        list_aux = []
        for i in range(self.number_cards):
            img = imageClass(self.data[i]['path'])
            img_anc = img.get_anchor()
            img_pos = img.get_positive()
            list_aux.append(self.__getSimilarRank(img_anc, img_anc)[0].item())
            list_aux.append(self.__getSimilarRank(img_anc, img_pos)[0].item())
            for j in range(self.number_cards):
                if i == j:
                    continue
                img_neg = imageClass(self.data[j]['path'])
                img_neg_anc = img_neg.get_anchor()
                list_aux.append(self.__getSimilarRank(img_anc, img_neg_anc)[0].item())
            list_ret.append(list_aux)
        return list_ret

    def start(self):
        img_card = imageClass(self.data[0]['path'])
        img_test = imageClass(self.data[1]['path'])
        print(self.__getSimilarRank(img_card, img_card)[0])
        print(self.__getSimilarRank(img_card, img_test)[0])


class SiameseNetwork(torch_nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.resnet.load_state_dict(torch.load('./training_results/12Cards50ephRESNET101.pth'), strict=False)

    def forward_once(self, input1):
        output = self.resnet(input1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
