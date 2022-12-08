import os
import random
import torch.nn as torch_nn
import torchvision.models as models
from torch import optim, save as save
from convertImage import ConvertImage as imageClass


class Train:
    def __init__(self, training_path, number_epochs=1, margin=1.0, learning_rate=0.05):
        # https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
        self.data = self.__load_data_paths(training_path)
        self.number_cards = len(self.data)
        self.number_epochs = number_epochs
        self.net = SiameseNetwork()
        self.loss_function = torch_nn.TripletMarginLoss(margin=margin)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.file_name = '10Cartas500epocasRESNET101'.format(number_epochs)
        self.train_log = open('./logs/' + self.file_name + '.txt', mode="a")

    def __load_data_paths(self, training_path):
        images_path = []
        for card in os.listdir(training_path):
            item = {
                'path': os.path.join(training_path, card),
                'card_id': card.strip('.jpg'),
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
                loss = self.loss_function(output1, output2, output3)
                loss.backward()
                self.optimizer.step()
                dist = (output1 - output2).pow(2).sum(1).pow(.5)
                print("Epoch number: {} Current card: {} ---> loss: {} ---> positive distance: {}".format(epoch, i, loss.item(), dist))
                print("Epoch number: {} Current card: {} ---> loss: {} ---> positive distance: {}".format(epoch, i, loss, dist), file=self.train_log)
            print()
            if (epoch + 1) % 10 == 0:
                file_name = '{}'.format(epoch + 1)
                save_path = './training_results/' + file_name + '.pth'
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
