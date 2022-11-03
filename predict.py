import os
import torch
import json
import torch.nn as torch_nn
import torchvision.models as models
import torch.nn.functional as nn_functional
from convertImage import ConvertImage as imageClass


class Predict:
    def __init__(self, predict_cards, dataset_cards, json_dict):
        self.net = SiameseNetwork()
        self.net.load_state_dict(torch.load('./training_results/10Cartas250epocasRESNET101.pth'))
        self.net.eval()
        self.data = self.__load_data_paths(predict_cards)
        self.dataset = self.__load_data_paths(dataset_cards)
        self.cards_details = self.__load_json(json_dict)
        self.number_cards_predict = len(self.data)
        self.number_cards_dataset = len(self.dataset)

    def __load_json(self, path):
        for jason_file in os.listdir(path):
            full_path = os.path.join(path, jason_file)
            file = open(full_path)
            data = json.load(file)
            file.close()
            return data['data']

    def __return_name(self, id):
        for card in self.cards_details:
            if id == str(card['id']):
                return card['name']
        return ''

    def __load_data_paths(self, cards_path):
        images_path = []
        for card in os.listdir(cards_path):
            item = {
                'path': os.path.join(cards_path, card),
                'name': card.strip('.jpg')
            }
            images_path.append(item)
        return images_path

    def __getSimilarRank(self, image, image2):
        output1, output2 = self.net(image, image2)
        euclidean_distance = nn_functional.pairwise_distance(output1, output2)
        return euclidean_distance

    def test_training(self):
        with torch.no_grad():
            for card in range(self.number_cards_predict):
                img_card = imageClass(self.data[card]['path']).get_anchor()
                for i in range(self.number_cards_dataset):
                    data = imageClass(self.dataset[i]['path']).get_anchor()
                    distance = self.__getSimilarRank(data, img_card).item()
                    name = ''
                    if distance <= 1:
                        name = self.__return_name(self.dataset[i]['name'])
                    else:
                        name = 'Nao encontrado'
                    print('{} -> {}: {} -- > {}'.format(self.dataset[i]['name'], self.data[card]['name'], distance, name))


class SiameseNetwork(torch_nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

    def forward_once(self, input1):
        output = self.resnet(input1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
