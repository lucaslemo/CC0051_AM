import torchvision.transforms as transforms
from PIL import Image as img
from random import randint


class ConvertImage:
    def __init__(self, image_path):
        image = img.open(image_path)
        image_crop = image.crop((50, 111, 370, 431))
        image_l = image_crop.convert("L")
        image_positive = self.__get_positive(image_l)
        self.image_ach = self.__gray_convert(image_l)
        self.image_pos = self.__gray_convert(image_positive)

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

    def __gray_convert(self, image):
        change_img = transforms.Compose([transforms.Grayscale(num_output_channels=3)])
        return change_img(image)

    def get_anchor(self):
        return self.image_ach

    def get_positive(self):
        return self.image_pos
