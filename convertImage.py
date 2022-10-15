import torchvision.transforms as transforms
from PIL import Image
from random import randint


class ConvertImage:
    def __init__(self, image_path):
        image = Image.open(image_path)
        image_crop = image.crop((50, 111, 370, 431))
        image_small = image_crop.resize((244, 244))
        self.image_l = image_small.convert("L")


    def __get_positive(self, image):
        rand1 = (randint(2, 4) / 8) + 0.25
        rand2 = (randint(2, 5) / 8) + 0.25
        rand3 = (randint(1, 5) / 8) + 0.25
        bright = (0.1, rand1 * 2.0)
        contrast = (0.5, rand2 * 1.5)
        saturation = (0.0, rand3 * 2.0)
        change_img = transforms.Compose(
            [transforms.ColorJitter(brightness=bright, contrast=contrast, saturation=saturation, hue=.5)]
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
