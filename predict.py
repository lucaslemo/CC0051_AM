import os
import torch
import json
import random
import torch.nn as torch_nn
import torchvision.models as models
import torch.nn.functional as nn_functional
from convertImage import ConvertImage as imageClass


class Predict:
    def __init__(self, predict_cards, dataset_cards, json_dict, model_path):
        self.data = self.__load_data_paths(predict_cards)
        self.dataset = self.__load_data_paths(dataset_cards)
        self.cards_details = self.__load_json(json_dict)
        self.model = model_path
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
        euclidean_distance = nn_functional.pairwise_distance(image, image2)
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
                print('')

    def model_test(self):
        with torch.no_grad():
            media = 0
            for card in range(self.number_cards_predict):
                img_card = imageClass(self.data[card]['path']).get_anchor()
                for i in range(self.number_cards_dataset):
                    if self.data[card]['name'] == self.dataset[i]['name']:
                        data = imageClass(self.dataset[i]['path']).get_anchor()
                        distance = self.__getSimilarRank(data, img_card).item()
                        media += distance
                        break
            media = media/self.number_cards_predict
            return media

    def start(self):
        net = SiameseNetwork()
        net.load_state_dict(torch.load(self.model))
        net.eval()
        with torch.no_grad():
            mean_positive = 0.0
            mean_negative = 0.0
            print(self.model)
            print(len(self.data))
            for card in range(len(self.data)):
                img_card_real = imageClass(self.data[card]['path']).get_anchor()
                while True:
                    img_card_database = random.choice(self.dataset)
                    if self.dataset[card]['name'] == img_card_database['name']:
                        data_card = imageClass(img_card_database['path']).get_anchor()
                        output1, output2 = net(data_card, img_card_real)
                        distance = self.__getSimilarRank(output1, output2).item()
                        print('{} --> {} Dist: {}'.format(self.dataset[card]['name'], img_card_database['name'], distance))
                        mean_positive += distance
                        break
                while True:
                    img_card_database = random.choice(self.dataset)
                    if self.dataset[card]['name'] != img_card_database['name']:
                        data_card = imageClass(img_card_database['path']).get_anchor()
                        output1, output2 = net(data_card, img_card_real)
                        distance = self.__getSimilarRank(output1, output2).item()
                        print('  {} --> {} Dist: {}'.format(self.dataset[card]['name'], img_card_database['name'], distance))
                        mean_negative += distance
                        break
            print()
            mean_positive = float(mean_positive / len(self.data))
            mean_negative = float(mean_negative / len(self.data))
            return mean_positive, mean_negative

    def teste_duas_cartas(self, image_path1, image_path2, model_path):
        net = SiameseNetwork()
        net.load_state_dict(torch.load(model_path))
        net.eval()
        image1 = imageClass(image_path1).get_anchor()
        image2 = imageClass(image_path2).get_anchor()
        with torch.no_grad():
            output1, output2 = net(image1, image2)
            distance = self.__getSimilarRank(output1, output2).item()
            return distance

    def testa_acertos(self):
        net = SiameseNetwork()
        net.load_state_dict(torch.load(self.model))
        net.eval()
        with torch.no_grad():
            qtd = 0
            for real_card in range(len(self.data)):
                min_distance = 100000
                match = -1
                img_card_real = imageClass(self.data[real_card]['path']).get_anchor()
                for data_card in range(len(self.dataset)):
                    img_card_data = imageClass(self.dataset[data_card]['path']).get_anchor()
                    output1, output2 = net(img_card_real, img_card_data)
                    distance = self.__getSimilarRank(output1, output2).item()
                    if distance < min_distance:
                        min_distance = distance
                        match = data_card
                if match != -1 and self.data[real_card]['name'] == self.dataset[match]['name']:
                    qtd = qtd + 1
            return qtd

    def calcula_todas_distancias(self):
        net = SiameseNetwork()
        net.load_state_dict(torch.load(self.model))
        net.eval()
        with torch.no_grad():
            print(self.model)
            result = []
            header = []
            header.append(self.model)
            for i in range(len(self.dataset)):
                header.append(self.dataset[i]['name'])
            result.append(header)
            for i in range(len(self.dataset)):
                img_card_real = imageClass(self.data[i]['path']).get_anchor()
                result_aux = []
                result_aux.append(self.dataset[i]['name'])
                for j in range(len(self.dataset)):
                    img_card_dataset = imageClass(self.dataset[j]['path']).get_anchor()
                    output1, output2 = net(img_card_real, img_card_dataset)
                    distance = self.__getSimilarRank(output1, output2)
                    distance = distance.item()
                    result_aux.append(distance)
                    print('{} --> {} Dist: {}'.format(self.dataset[i]['name'], self.dataset[j]['name'], distance))
                print()
                result.append(result_aux)
            return result


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
