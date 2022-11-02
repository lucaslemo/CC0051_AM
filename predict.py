import os
import torch
import torch.nn as torch_nn
import torchvision.models as models
import torch.nn.functional as nn_functional
from convertImage import ConvertImage as imageClass


class Predict:
    def __init__(self, cards_path, model_path):
        self.net = SiameseNetwork(model_path)
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


class SiameseNetwork(torch_nn.Module):
    def __init__(self, model_path):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.resnet.load_state_dict(torch.load(model_path), strict=False)
        self.resnet.eval()

    def forward_once(self, input1):
        output = self.resnet(input1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
