import os
import random
import torch.nn as torch_nn
import torchvision.models as models
import torch.nn.functional as nn_functional
from torch import optim, save as save
from convertImage import ConvertImage as imageClass


class Train:
    def __init__(self, training_path, batch_size=48, number_epochs=300, margin=2.0, learning_rate=0.0005):
        # https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
        self.data = self.__load_data_paths(training_path)
        self.number_cards = len(self.data)
        self.batch_size = batch_size
        self.number_epochs = number_epochs
        self.net = torch_nn.DataParallel(SiameseNetwork(), device_ids=[0, 1, 2, 3])
        self.criterion = TripletLoss(margin)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.train_log = open('./logs/12Cards50ephRESNET101.txt', mode="a")

    def __load_data_paths(self, training_path):
        images_path = []
        for card in os.listdir(training_path):
            item = {
                'path': os.path.join(training_path, card),
                'card_id': int(card.strip('.jpg')),
                'status_use': 0
            }
            images_path.append(item)
        return images_path

    def __chose_cards(self, epoch):
        rand_acn = random.randint(0, self.number_cards - 1)
        while self.data[rand_acn]['status_use'] > epoch:
            if rand_acn == self.number_cards - 1:
                rand_acn = 0
            else:
                rand_acn += 1
        rand_neg = random.randint(0, self.number_cards - 1)
        while self.data[rand_neg]['card_id'] == self.data[rand_acn]['card_id']:
            if rand_neg == self.number_cards - 1:
                rand_neg = 0
            else:
                rand_neg += 1
        self.data[rand_acn]['status_use'] += 1
        img = imageClass(self.data[rand_acn]['path'])
        neg = imageClass(self.data[rand_neg]['path'])
        return img.get_anchor(), img.get_positive(), neg.get_anchor()

    def start(self):
        for epoch in range(self.number_epochs):
            for i in range(self.number_cards):
                img_anc, img_pos, img_neg = self.__chose_cards(epoch)
                self.optimizer.zero_grad()
                output1, output2, output3 = self.net(img_anc, img_pos, img_neg)
                loss_contrastive = self.criterion(output1, output2, output3)
                loss_contrastive.backward()
                self.optimizer.step()
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()), file=self.train_log)
                print("Epoch number {}\n Current card {}\n".format(epoch, i))
            save_path = './training_results/12Cards50ephRESNET101.pth'
            save(self.net.state_dict(), save_path)

    def __del__(self):
        self.train_log.close()


# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# https://pytorch.org/vision/main/models.html
# https://pytorch.org/hub/pytorch_vision_resnet/
class SiameseNetwork(torch_nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

    def forward_once(self, input1):
        output = self.resnet(input1)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3


class TripletLoss(torch_nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = nn_functional.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
