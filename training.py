import torch
import torch.nn as torch_nn
import torchvision.models as models
import torch.nn.functional as nn_functional
import torchvision.transforms as transforms
import torchvision.datasets as dataset
from torch import optim
from convertImage import ConvertImage as imageClass


class Train:
    def __init__(self, training_path, batch_size=48, number_epochs=300, margin=2.0, learning_rate=0.0005):
        # https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
        self.training_path = training_path
        self.batch_size = batch_size
        self.number_epochs = number_epochs
        self.net = torch_nn.DataParallel(SiameseNetwork().cuda(), device_ids=[0])
        self.criterion = TripletLoss(margin)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

    def __prepare_images(self, img):
        # https://pytorch.org/hub/pytorch_vision_resnet/
        # https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
        # https://pytorch.org/docs/stable/tensors.html#:~:text=A%20torch.Tensor%20is%20a,of%20a%20single%20data%20type.
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]
        )
        return transform(img).unsqueeze(0)

    def start(self):
        for epoch in range(self.number_epochs):
            for i in range(len(self.training_path)):
                pass
                '''
                self.optimizer.zero_grad()
                output1, output2, output3 = net(img_anc, img_pos, img_neg)
                loss_contrastive = self.criterion(output1, output2, output3)
                loss_contrastive.backward()
                self.optimizer.step()
                save_path = './test.pth'
                torch.save(self.net.state_dict(), save_path)
                '''


# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# https://pytorch.org/vision/main/models.html
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


# http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
class ContrastiveLoss(torch_nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn_functional.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class TripletLoss(torch_nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = nn_functional.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
