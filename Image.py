import torchvision.transforms as transforms
from PIL import Image as img
from random import randint


class Image:
    def __init__(self, image_path):
        image = img.open(image_path)
        image_crop = image.crop((50, 111, 370, 431))
        image_l = image_crop.convert("L")
        image_positive = self.__get_positive(image_l)
        self.image_ach = self.__gray_convert(image_l)
        self.image_pos = self.__gray_convert(image_positive)

    def __get_positive(self, image):
        rand1 = (randint(1, 10) / 10) * 2
        rand2 = (randint(1, 10) / 10) * 2
        bright = (rand1, rand1*1.9)
        contrast = (rand2, rand2*1.9)
        change_img = transforms.Compose(
            [transforms.ColorJitter(brightness=bright, contrast=contrast, hue=.05, saturation=(.0, .15))]
        )
        return change_img(image)

    def __gray_convert(self, image):
        change_img = transforms.Compose([transforms.Grayscale(num_output_channels=3)])
        return change_img(image)

    def get_anchor(self):
        return self.image_ach

    def get_positive(self):
        return self.image_pos
