import os
import torch
import json
import cv2
import random
import numpy as np
import torch.nn as torch_nn
import models.resnet as mod_res
import torch.nn.functional as nn_functional
from torchvision import models
from matplotlib import pyplot as plt
from torch.autograd import Variable


class Predict:
    def __init__(self, real_cards, dataset_cards, model):
        self.test_dataset = self.__load_data_paths(real_cards)
        self.dataset = self.__load_data_paths(dataset_cards)
        self.model = model
        

    def __load_json(self, path):
        for jason_file in os.listdir(path):
            full_path = os.path.join(path, jason_file)
            file = open(full_path)
            data = json.load(file)
            file.close()
            return data['data']

    def __load_data_paths(self, cards_path):
        images_path = []
        for card in os.listdir(cards_path):
            item = {
                'path': os.path.join(cards_path, card),
                'name': card.split('.')[0]
            }
            images_path.append(item)
        return images_path
    
    def __imshow(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path, 0)
        img1 = cv2.resize(img1, (421, 614), interpolation=cv2.INTER_AREA)
        img1 = img1[111:431, 50:370]
        img1 = cv2.resize(img1, (244, 244), interpolation=cv2.INTER_AREA)

        img2 = cv2.imread(img2_path, 0)
        img2 = cv2.resize(img2, (421, 614), interpolation=cv2.INTER_AREA)
        img2 = img2[111:431, 50:370]
        img2 = cv2.resize(img2, (244, 244), interpolation=cv2.INTER_AREA)
        im_h = cv2.hconcat([img1, img2])

        cv2.imshow('image.jpeg', im_h)
        cv2.waitKey(0)

    def __cv_prepare(self, image_path):
        img = cv2.imread(image_path, 0)
        img = img[111:431, 50:370]
        img = cv2.resize(img, (244, 244), interpolation=cv2.INTER_AREA)
        img = cv2.equalizeHist(img)
        img = [img] * 3
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).type('torch.FloatTensor')
        return img

    def __load_img(self, path):
        for img_file in os.listdir(path):
            image_path = os.path.join(path, img_file)
            return self.__cv_prepare(image_path)

    def __getSimilarRank(self, output1, output2):
        euclidean_distance = nn_functional.pairwise_distance(output1, output2)
        return euclidean_distance

    def start(self):
        net = SiameseNetwork()
        net.load_state_dict(torch.load(self.model))
        net.eval()
        with torch.no_grad():
            mean_positive = 0.0
            mean_negative = 0.0
            print(self.model)
            for card in range(len(self.test_dataset)):
                img_card_real = self.__cv_prepare(self.test_dataset[card]['path'])
                while True:
                    img_card_database = random.choice(self.dataset)
                    if self.test_dataset[card]['name'] == img_card_database['name']:
                        data_card = self.__cv_prepare(img_card_database['path'])
                        output1, output2 = net(Variable(data_card), Variable(img_card_real))
                        distance = self.__getSimilarRank(output1, output2).item()
                        self.__imshow(img_card_database['path'], self.test_dataset[card]['path'])
                        print('{} --> {} Dist: {}'.format(self.test_dataset[card]['name'], img_card_database['name'], distance))
                        mean_positive += distance
                        break
                while True:
                    img_card_database = random.choice(self.dataset)
                    if self.test_dataset[card]['name'] != img_card_database['name']:
                        data_card = self.__cv_prepare(img_card_database['path'])
                        output1, output2 = net(Variable(data_card), Variable(img_card_real))
                        distance = self.__getSimilarRank(output1, output2).item()
                        print('  {} --> {} Dist: {}'.format(self.test_dataset[card]['name'], img_card_database['name'], distance))
                        mean_negative += distance
                        break
            print()
            mean_positive = float(mean_positive / len(self.test_dataset))
            mean_negative = float(mean_negative / len(self.test_dataset))
            return mean_positive, mean_negative


class SiameseNetwork(torch_nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.resnet = mod_res.resnet101(filter_size=3)

    def forward_once(self, input1):
        output = self.resnet(input1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
