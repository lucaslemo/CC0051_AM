import torchvision.transforms as transforms
from PIL import Image
from random import randint


class ConvertImage:
    def __init__(self, image_path):
        image = Image.open(image_path)
        image_full = image.resize((421, 614))
        image_crop = image_full.crop((50, 111, 370, 431))
        image_small = image_crop.resize((244, 244))
        self.image_l = image_small.convert("L")

    def __get_positive(self, image):
        change_img = transforms.Compose(
            [
                transforms.ColorJitter(brightness=(0.5,1.5),contrast=(0.3,2.0),hue=.05, saturation=(.0,.15)),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0,0.3), scale=(0.6,1.8), shear=(0.0,0.4), interpolation=transforms.InterpolationMode.NEAREST, fill=0)
            ]
        )
        return change_img(image)

    def __transform(self, image):
        # https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
        # https://pytorch.org/docs/stable/tensors.html#:~:text=A%20torch.Tensor%20is%20a,of%20a%20single%20data%20type.
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ]
        )
        return transform(image).unsqueeze(0)

    def get_anchor(self):
        return self.__transform(self.image_l)

    def get_positive(self):
        positive = self.__get_positive(self.image_l)
        return self.__transform(positive)
