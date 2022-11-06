import os
import random
import PIL
import models.resnet as mod_res
import torch.nn as torch_nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch import optim, save as save


class Train:
    def __init__(self, training_path, number_epochs, batch_size, margin, learning_rate):
        self.folder_dataset = dset.ImageFolder(root=training_path)
        self.dataset = CustomDataset(
            imageFolderDataset=self.folder_dataset,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize((244,244)),
                    transforms.ColorJitter(
                        brightness=(0.2,1.5),
                        contrast=(0.1,2.5),
                        hue=.05,
                        saturation=(.0,.15)
                    ),
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(
                        0, 
                        translate=(0,0.3),
                        scale=(0.6,1.8),
                        shear=(0.0,0.4),
                        interpolation=transforms.InterpolationMode.NEAREST,
                        fill = 0
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            ),
            should_invert=False
        )
        self.train_dataloader = DataLoader(
            self.dataset,
            shuffle=True,
            num_workers=4,
            batch_size=batch_size
        )
        self.net = SiameseNetwork()
        self.loss = torch_nn.TripletMarginLoss(margin=margin)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.number_epochs = number_epochs
    
    def start(self):
        counter = []
        loss_history = [] 
        iteration_number= 0
        prevNum = -1
        for epoch in range(self.number_epochs):
            for i, data in enumerate(self.train_dataloader, 0):
                img_anc, img_pos, img_neg, _ = data
                self.optimizer.zero_grad()
                output1, output2, output3 = self.net(img_anc, img_pos , img_neg)
                loss_contrastive = self.loss(output1, output2, output3)
                loss_contrastive.backward()
                self.optimizer.step()
                # To prevent repetation of epoch
                if i % 10 == 0 and prevNum != epoch:
                    print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                    iteration_number +=10
                    counter.append(iteration_number)
                    loss_history.append(loss_contrastive.item())
                    prevNum = epoch
            if (epoch + 1) % 10 == 0:
                save_path = './training_results/v3/' + '{}'.format(epoch + 1) + '.pth'
                save(self.net.state_dict(), save_path)    


class CustomDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # Get an image from the same class
        while True:
            #keep looping till the same class image is found
            img1_tuple = random.choice(self.imageFolderDataset.imgs) 
            if img0_tuple[1] == img1_tuple[1]:
                break

        # Get an image from a different class
        while True:
            #keep looping till a different class image is found
            img2_tuple = random.choice(self.imageFolderDataset.imgs) 
            if img0_tuple[1] != img2_tuple[1]:
                break

        width, height = (244,244)

        pathList = []
        pathList.append((img0_tuple[0], img1_tuple[0], img2_tuple[0]))

        # Open imgs
        img0 = PIL.Image.open(img0_tuple[0])
        img1 = PIL.Image.open(img1_tuple[0])
        img2 = PIL.Image.open(img2_tuple[0])

        # Crop the card art
        img0 = img0.crop((50, 111, 370, 431))
        img1 = img1.crop((50, 111, 370, 431))
        img2 = img2.crop((50, 111, 370, 431))

        # Resize card art
        img0 = img0.resize((width, height))
        img1 = img1.resize((width, height))
        img2 = img2.resize((width, height))
        
        # Convert for L mode
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        img2 = img2.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)
            img2 = PIL.ImageOps.invert(img2)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # anchor, positive image, negative image
        return img0, img1, img2, pathList

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetwork(torch_nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.resnet = mod_res.resnet101(filter_size=3)

    def forward_once(self, input1):
        output = self.resnet(input1)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3
